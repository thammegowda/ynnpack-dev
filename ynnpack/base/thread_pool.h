// Licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_THREAD_POOL_H_
#define XNNPACK_YNNPACK_BASE_THREAD_POOL_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "ynnpack/include/ynnpack.h"

namespace ynn {

inline constexpr std::size_t CACHE_LINE_BYTES = 64;

/// Lock for few-instruction critical sections (exception capture).
class SpinLock {
public:
    void lock() noexcept;
    bool try_lock() noexcept;
    void unlock() noexcept;

private:
    std::atomic<bool> locked_ = false;
};

/// Per-worker execution context passed into parallel range and multi-phase team callbacks.
struct WorkerContext {
    std::size_t tid = 0;          // [0, thread_count); 0 is the calling thread
    std::size_t thread_count = 1; // active team size for this dispatch
    std::byte* scratch = nullptr; // cache-line-aligned persistent per-worker scratchpad
    std::size_t scratch_size = 0;
};

/// Sense-reversing user-space barrier synchronizing all participants of a `WorkerPool::run_team` launch.
class PhaseBarrier {
public:
    explicit PhaseBarrier(std::size_t participants = 1) noexcept;

    void reset(std::size_t participants) noexcept;
    void abort() noexcept;
    bool aborted() const noexcept { return aborted_.load(std::memory_order_acquire); }
    bool arrive_and_wait() noexcept;

private:
    std::size_t participants_ = 1;
    std::atomic<bool> aborted_ = false;
    alignas(CACHE_LINE_BYTES) std::atomic<std::size_t> remaining_ = 1;
    alignas(CACHE_LINE_BYTES) std::atomic<std::uint64_t> phase_ = 0;
};

/// Multi-phase range-stealing slot array for use inside `WorkerPool::run_team`.
struct alignas(CACHE_LINE_BYTES) ParallelRangeSlot {
    std::atomic<std::size_t> next = 0;
    std::size_t end = 0;
};

class ParallelRangeSlots {
public:
    static constexpr std::size_t INLINE_CAP = 32;

    ParallelRangeSlots(std::size_t items, std::size_t workers) : count_(workers) {
        if (workers > INLINE_CAP) heap_ = std::make_unique<ParallelRangeSlot[]>(workers);
        auto* s = data();
        for (std::size_t w = 0; w < workers; ++w) {
            s[w].next.store((items * w) / workers, std::memory_order_relaxed);
            s[w].end = (items * (w + 1)) / workers;
        }
    }

    ParallelRangeSlot* data() noexcept { return heap_ ? heap_.get() : inline_.data(); }
    std::size_t size() const noexcept { return count_; }

private:
    std::size_t count_ = 0;
    std::array<ParallelRangeSlot, INLINE_CAP> inline_{};
    std::unique_ptr<ParallelRangeSlot[]> heap_;
};

template <typename Fn>
void drain_range_slots(ParallelRangeSlots& slots, std::size_t tid, std::size_t grain, Fn&& fn) {
    const auto count = slots.size();
    auto* arr = slots.data();
    for (std::size_t offset = 0; offset < count; ++offset) {
        auto& slot = arr[(tid + offset) % count];
        const auto limit = slot.end;
        for (;;) {
            auto cur = slot.next.load(std::memory_order_relaxed);
            if (cur >= limit) break;
            const auto next = std::min(limit, cur + grain);
            if (slot.next.compare_exchange_weak(cur, next, std::memory_order_relaxed)) fn(cur, next);
        }
    }
}

/// Low-overhead CPU thread pool for YNNPACK composites and host inference runtimes.
///
/// Provides:
/// - Explicit `WorkerContext` (`tid`, `thread_count`, and persistent cache-aligned per-worker scratchpads).
/// - Deterministic per-worker home-range slots with contention-free ring stealing (`run_ranges`).
/// - Single-dispatch multi-phase team execution (`run_team` + `PhaseBarrier`) so multi-stage ops synchronize in L1/L2
///   without re-entering the thread pool.
/// - Lock-free fork/join: one published cache line per dispatch and per-worker completion flags.
/// - Hardware-timed `yield` spinning + C++20 `std::atomic::wait` (`ulock`/`futex`) parking with zero mutex wakeups.
class WorkerPool final {
public:
    using TeamFn = void (*)(void* context, const WorkerContext& worker, PhaseBarrier& barrier);
    using RangeFn = void (*)(void* context, const WorkerContext& worker, std::size_t begin, std::size_t end);
    using TaskFn = void (*)(void* context, std::size_t task);

    explicit WorkerPool(int workers);
    ~WorkerPool();

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    int thread_count() const noexcept { return workers_; }
    std::size_t total_threads() const noexcept { return static_cast<std::size_t>(workers_ > 0 ? workers_ + 1 : 1); }

    void run_team(std::size_t max_threads, std::size_t scratch_bytes, void* context, TeamFn fn);
    void run_ranges(std::size_t items, std::size_t min_grain, std::size_t scratch_bytes, void* context, RangeFn fn);
    void run_tasks(std::size_t n, void* context, TaskFn fn);
    void parallel_for(std::size_t n, const std::function<void(std::size_t)>& body);

private:
    struct ScratchBuffer {
        std::byte* data = nullptr;
        std::size_t capacity = 0;

        ScratchBuffer() = default;
        ~ScratchBuffer();
        ScratchBuffer(ScratchBuffer&& other) noexcept;
        ScratchBuffer& operator=(ScratchBuffer&& other) noexcept;
        ScratchBuffer(const ScratchBuffer&) = delete;
        ScratchBuffer& operator=(const ScratchBuffer&) = delete;

        void ensure(std::size_t required_bytes);
    };

    struct alignas(CACHE_LINE_BYTES) RangeSlot {
        std::atomic<std::size_t> next = 0;
        std::size_t end = 0;
        std::size_t home_grain = 1;
        std::size_t steal_grain = 1;
    };

    struct RangePayload {
        RangeSlot* slots = nullptr;
        void* user_ctx = nullptr;
        RangeFn user_fn = nullptr;
    };

    // Everything a worker needs to join a dispatch shares one cache line and is published by the release store of
    // `token` ((seq << 16) | team size), so joining costs a single cache miss. The caller rewrites the line only after
    // every member has reported done.
    struct alignas(CACHE_LINE_BYTES) TeamLine {
        std::atomic<std::uint64_t> token = 0;
        TeamFn fn = nullptr;
        void* ctx = nullptr;
        std::size_t scratch_bytes = 0;
        RangePayload range;
    };

    // Each worker reports completion on its own line, so the join has no shared read-modify-write.
    struct alignas(CACHE_LINE_BYTES) DoneSlot {
        std::atomic<std::uint64_t> seq = 0;
    };

    void dispatch(std::size_t team_size, TeamFn fn, void* context, std::size_t scratch_bytes);
    void execute_team_member(std::size_t tid, std::size_t team_size) noexcept;
    void record_exception(std::exception_ptr error) noexcept;
    void worker_main(std::size_t tid);
    void idle_worker(std::uint64_t seen, std::uint64_t last_team_seq);

    int workers_;
    std::vector<std::thread> threads_;
    std::vector<ScratchBuffer> scratches_;
    std::unique_ptr<RangeSlot[]> range_slots_;
    std::unique_ptr<DoneSlot[]> done_;
    std::atomic<bool> stop_ = false;

    alignas(CACHE_LINE_BYTES) SpinLock caller_lock_;
    std::uint64_t team_seq_ = 0;
    SpinLock error_lock_;
    std::exception_ptr team_error_;
    PhaseBarrier team_barrier_;

    TeamLine team_;

    // Bumped only to wake parked workers; spinning workers watch `team_.token`.
    alignas(CACHE_LINE_BYTES) std::atomic<std::uint64_t> event_seq_ = 0;
    alignas(CACHE_LINE_BYTES) std::atomic<int> sleeping_ = 0;
};

void set_thread_count(std::size_t total_threads) noexcept;
std::size_t thread_count() noexcept;
ynn_status reserve_thread_pool(std::size_t total_threads);
ynn_status parallel_for(std::size_t size, const std::function<void(std::size_t)>& body);
ynn_status parallel_tasks(std::size_t tasks, void* context, void (*body)(void*, std::size_t));
ynn_status parallel_ranges(std::size_t items, std::size_t min_grain, void* context,
                           void (*body)(void*, std::size_t, std::size_t));
ynn_status parallel_ranges_with_worker(std::size_t items, std::size_t min_grain, std::size_t scratch_bytes,
                                       void* context,
                                       void (*body)(void*, const WorkerContext&, std::size_t, std::size_t));
ynn_status parallel_team(std::size_t max_threads, std::size_t scratch_bytes, void* context,
                         void (*body)(void*, const WorkerContext&, PhaseBarrier&));

template <typename Body>
ynn_status parallel_tasks(std::size_t tasks, Body&& body) {
    using Callable = std::remove_reference_t<Body>;
    return parallel_tasks(tasks, const_cast<void*>(static_cast<const void*>(std::addressof(body))),
                          [](void* context, std::size_t task) { (*static_cast<Callable*>(context))(task); });
}

template <typename Body>
ynn_status parallel_ranges(std::size_t items, std::size_t min_grain, Body&& body) {
    using Callable = std::remove_reference_t<Body>;
    return parallel_ranges(
        items, min_grain, const_cast<void*>(static_cast<const void*>(std::addressof(body))),
        [](void* context, std::size_t begin, std::size_t end) { (*static_cast<Callable*>(context))(begin, end); });
}

template <typename Body>
ynn_status parallel_ranges_with_worker(std::size_t items, std::size_t min_grain, std::size_t scratch_bytes,
                                       Body&& body) {
    using Callable = std::remove_reference_t<Body>;
    return parallel_ranges_with_worker(items, min_grain, scratch_bytes,
                                       const_cast<void*>(static_cast<const void*>(std::addressof(body))),
                                       [](void* context, const WorkerContext& worker, std::size_t begin,
                                          std::size_t end) { (*static_cast<Callable*>(context))(worker, begin, end); });
}

template <typename Body>
ynn_status parallel_team(std::size_t max_threads, std::size_t scratch_bytes, Body&& body) {
    using Callable = std::remove_reference_t<Body>;
    return parallel_team(max_threads, scratch_bytes, const_cast<void*>(static_cast<const void*>(std::addressof(body))),
                         [](void* context, const WorkerContext& worker, PhaseBarrier& barrier) {
                             (*static_cast<Callable*>(context))(worker, barrier);
                         });
}

std::uint64_t supported_arch_flags() noexcept;
std::string supported_arch_names();

} // namespace ynn

#endif // XNNPACK_YNNPACK_BASE_THREAD_POOL_H_
