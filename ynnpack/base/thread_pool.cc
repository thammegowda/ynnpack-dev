// Licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.

#include "ynnpack/base/thread_pool.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <limits>
#include <mutex>
#include <new>
#include <unordered_map>
#include <utility>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/topology.h"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

namespace ynn {
namespace {

constexpr int LOCK_PAUSES = 128;
constexpr int BARRIER_PAUSES = 4096;
constexpr std::uint64_t SPIN_MICROSECONDS = 180;
constexpr int GENERIC_SPIN_YIELDS = 64;

inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    _mm_pause();
#elif (defined(__aarch64__) || defined(__arm__)) && (defined(__GNUC__) || defined(__clang__))
    __asm__ __volatile__("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

#if (defined(__aarch64__) || defined(__arm64__)) && (defined(__GNUC__) || defined(__clang__))
inline std::uint64_t arm_counter_ticks() noexcept {
    std::uint64_t ticks = 0;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(ticks));
    return ticks;
}

inline std::uint64_t arm_spin_budget_ticks(std::uint64_t microseconds) noexcept {
    static const std::uint64_t freq = [] {
        std::uint64_t f = 0;
        __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(f));
        return f ? f : std::uint64_t{24'000'000};
    }();
    return std::max<std::uint64_t>(1, (freq * microseconds) / 1'000'000);
}
#endif

template <typename Ready>
bool spin_until_ready(Ready&& ready) noexcept {
    if (ready()) return true;
#if (defined(__aarch64__) || defined(__arm64__)) && (defined(__GNUC__) || defined(__clang__))
    const auto budget = arm_spin_budget_ticks(SPIN_MICROSECONDS);
    const auto start = arm_counter_ticks();
    do {
        for (int i = 0; i < 16; ++i) {
            if (ready()) return true;
            cpu_relax();
        }
    } while (arm_counter_ticks() - start < budget);
#elif defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    // PAUSE latency varies ~10x across x86 generations, so bound the spin by time rather than a pause count.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(SPIN_MICROSECONDS);
    do {
        for (int i = 0; i < 64; ++i) {
            if (ready()) return true;
            _mm_pause();
        }
    } while (std::chrono::steady_clock::now() < deadline);
#else
    for (int i = 0; i < GENERIC_SPIN_YIELDS; ++i) {
        if (ready()) return true;
        std::this_thread::yield();
    }
#endif
    return ready();
}

thread_local const WorkerPool* active_team_pool = nullptr;

class ActiveTeamScope {
public:
    explicit ActiveTeamScope(const WorkerPool* pool) noexcept : previous_(std::exchange(active_team_pool, pool)) {}
    ~ActiveTeamScope() { active_team_pool = previous_; }
    ActiveTeamScope(const ActiveTeamScope&) = delete;
    ActiveTeamScope& operator=(const ActiveTeamScope&) = delete;

private:
    const WorkerPool* previous_;
};

struct InlineScratchStack {
    std::vector<std::unique_ptr<std::byte[], void (*)(std::byte*)>> slots;
    std::vector<std::size_t> capacities;
    std::size_t depth = 0;
};

thread_local InlineScratchStack tls_inline_scratch;

class InlineScratchScope {
public:
    explicit InlineScratchScope(std::size_t required_bytes) {
        if (required_bytes == 0) return;
        auto& stack = tls_inline_scratch;
        index_ = stack.depth++;
        active_ = true;
        if (index_ >= stack.slots.size()) {
            stack.slots.emplace_back(nullptr, [](std::byte* ptr) {
                if (ptr != nullptr) ::operator delete(ptr, std::align_val_t{CACHE_LINE_BYTES});
            });
            stack.capacities.push_back(0);
        }
        if (stack.capacities[index_] < required_bytes) {
            const auto alloc_bytes = std::max(required_bytes, stack.capacities[index_] * 2);
            auto* fresh = static_cast<std::byte*>(::operator new(alloc_bytes, std::align_val_t{CACHE_LINE_BYTES}));
            stack.slots[index_].reset(fresh);
            stack.capacities[index_] = alloc_bytes;
        }
        data_ = stack.slots[index_].get();
    }

    ~InlineScratchScope() {
        if (active_) --tls_inline_scratch.depth;
    }

    InlineScratchScope(const InlineScratchScope&) = delete;
    InlineScratchScope& operator=(const InlineScratchScope&) = delete;

    std::byte* data() const noexcept { return data_; }

private:
    std::byte* data_ = nullptr;
    std::size_t index_ = 0;
    bool active_ = false;
};

std::atomic<std::size_t> configured_thread_count = 0;

struct CachedPool {
    std::size_t threads = 0;
    std::shared_ptr<WorkerPool> pool;
};

void ensure_interactive_qos() noexcept {
#if defined(__APPLE__)
    thread_local bool configured = false;
    if (!configured) {
        if (::pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) != 0)
            (void)::pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
        configured = true;
    }
#endif
}

WorkerPool* borrow_thread_pool(std::size_t total_threads) {
    struct PoolCache {
        std::mutex mutex;
        std::unordered_map<std::size_t, std::shared_ptr<WorkerPool>> pools;
    };
    static PoolCache cache;

    if (total_threads == 0) total_threads = thread_count();
    if (total_threads == 0 || total_threads > static_cast<std::size_t>(std::numeric_limits<int>::max())) return nullptr;

    ensure_interactive_qos();
    thread_local CachedPool local;
    if (local.pool && local.threads == total_threads) return local.pool.get();

    std::scoped_lock lock(cache.mutex);
    if (auto found = cache.pools.find(total_threads); found != cache.pools.end()) {
        if (total_threads > 1 || !local.pool) local = {total_threads, found->second};
        return found->second.get();
    }
    auto created = std::make_shared<WorkerPool>(static_cast<int>(total_threads - 1));
    auto* raw = created.get();
    cache.pools.emplace(total_threads, created);
    if (total_threads > 1 || !local.pool) local = {total_threads, std::move(created)};
    return raw;
}

} // namespace

void SpinLock::lock() noexcept {
    int spins = 0;
    while (locked_.exchange(true, std::memory_order_acquire)) {
        while (locked_.load(std::memory_order_relaxed)) {
            if (spins < LOCK_PAUSES) {
                ++spins;
                cpu_relax();
            } else {
                std::this_thread::yield();
            }
        }
    }
}

bool SpinLock::try_lock() noexcept {
    return !locked_.load(std::memory_order_relaxed) && !locked_.exchange(true, std::memory_order_acquire);
}

void SpinLock::unlock() noexcept { locked_.store(false, std::memory_order_release); }

PhaseBarrier::PhaseBarrier(std::size_t participants) noexcept
    : participants_(std::max<std::size_t>(1, participants)), remaining_(participants_) {}

void PhaseBarrier::reset(std::size_t participants) noexcept {
    participants_ = std::max<std::size_t>(1, participants);
    aborted_.store(false, std::memory_order_relaxed);
    remaining_.store(participants_, std::memory_order_relaxed);
    phase_.store(0, std::memory_order_relaxed);
}

void PhaseBarrier::abort() noexcept {
    aborted_.store(true, std::memory_order_release);
    phase_.fetch_add(1, std::memory_order_release);
}

bool PhaseBarrier::arrive_and_wait() noexcept {
    if (aborted_.load(std::memory_order_acquire)) return false;
    if (participants_ <= 1) return true;
    const auto observed = phase_.load(std::memory_order_acquire);
    if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        remaining_.store(participants_, std::memory_order_relaxed);
        phase_.fetch_add(1, std::memory_order_release);
        return !aborted_.load(std::memory_order_acquire);
    }
    int spins = 0;
    while (phase_.load(std::memory_order_acquire) == observed && !aborted_.load(std::memory_order_acquire)) {
        if (spins < BARRIER_PAUSES) {
            ++spins;
            cpu_relax();
        } else {
            std::this_thread::yield();
        }
    }
    return !aborted_.load(std::memory_order_acquire);
}

WorkerPool::ScratchBuffer::~ScratchBuffer() {
    if (data != nullptr) ::operator delete(data, std::align_val_t{CACHE_LINE_BYTES});
}

WorkerPool::ScratchBuffer::ScratchBuffer(ScratchBuffer&& other) noexcept
    : data(std::exchange(other.data, nullptr)), capacity(std::exchange(other.capacity, 0)) {}

WorkerPool::ScratchBuffer& WorkerPool::ScratchBuffer::operator=(ScratchBuffer&& other) noexcept {
    if (this != &other) {
        if (data != nullptr) ::operator delete(data, std::align_val_t{CACHE_LINE_BYTES});
        data = std::exchange(other.data, nullptr);
        capacity = std::exchange(other.capacity, 0);
    }
    return *this;
}

void WorkerPool::ScratchBuffer::ensure(std::size_t required_bytes) {
    if (required_bytes <= capacity) return;
    const auto alloc_bytes = std::max(required_bytes, capacity * 2);
    auto* fresh = static_cast<std::byte*>(::operator new(alloc_bytes, std::align_val_t{CACHE_LINE_BYTES}));
    if (data != nullptr) ::operator delete(data, std::align_val_t{CACHE_LINE_BYTES});
    data = fresh;
    capacity = alloc_bytes;
}

WorkerPool::WorkerPool(int workers) : workers_(std::max(workers, 0)) {
    (void)get_supported_arch_flags();
    const auto total = total_threads();
    scratches_.resize(total);
    range_slots_ = std::make_unique<RangeSlot[]>(total);
    done_ = std::make_unique<DoneSlot[]>(total);
    threads_.reserve(static_cast<std::size_t>(workers_));
    for (int i = 0; i < workers_; ++i) {
        const auto tid = static_cast<std::size_t>(i + 1);
        threads_.emplace_back([this, tid] { worker_main(tid); });
    }
}

WorkerPool::~WorkerPool() {
    stop_.store(true, std::memory_order_release);
    event_seq_.fetch_add(1, std::memory_order_seq_cst);
    event_seq_.notify_all();
    for (auto& thread : threads_) thread.join();
}

void WorkerPool::record_exception(std::exception_ptr error) noexcept {
    {
        std::scoped_lock lock(error_lock_);
        if (!team_error_) team_error_ = std::move(error);
    }
    team_barrier_.abort();
}

void WorkerPool::execute_team_member(std::size_t tid, std::size_t team_size) noexcept {
    const ActiveTeamScope scope(this);
    const WorkerContext worker{
        .tid = tid,
        .thread_count = team_size,
        .scratch = team_.scratch_bytes > 0 ? scratches_[tid].data : nullptr,
        .scratch_size = team_.scratch_bytes,
    };
    try {
        team_.fn(team_.ctx, worker, team_barrier_);
    } catch (...) {
        record_exception(std::current_exception());
    }
}

void WorkerPool::run_team(std::size_t max_threads, std::size_t scratch_bytes, void* context, TeamFn fn) {
    if (max_threads == 0) return;
    const auto team_size = std::min(max_threads, total_threads());

    if (team_size <= 1 || active_team_pool == this) {
        const InlineScratchScope inline_scratch(scratch_bytes);
        PhaseBarrier single_barrier(1);
        const WorkerContext worker{
            .tid = 0,
            .thread_count = 1,
            .scratch = inline_scratch.data(),
            .scratch_size = scratch_bytes,
        };
        fn(context, worker, single_barrier);
        return;
    }

    std::scoped_lock caller_guard(caller_lock_);

    if (scratch_bytes > 0) {
        for (std::size_t t = 0; t < team_size; ++t) scratches_[t].ensure(scratch_bytes);
    }

    team_barrier_.reset(team_size);
    dispatch(team_size, fn, context, scratch_bytes);
}

void WorkerPool::dispatch(std::size_t team_size, TeamFn fn, void* context, std::size_t scratch_bytes) {
    team_.fn = fn;
    team_.ctx = context;
    team_.scratch_bytes = scratch_bytes;
    team_error_ = nullptr;

    const auto seq = ++team_seq_;
    // seq_cst pairs with the parking protocol in `idle_worker`: either a parking worker sees the new token, or this
    // thread sees it in `sleeping_` and wakes it.
    team_.token.store((seq << 16) | static_cast<std::uint64_t>(team_size), std::memory_order_seq_cst);
    if (sleeping_.load(std::memory_order_seq_cst) > 0) {
        event_seq_.fetch_add(1, std::memory_order_seq_cst);
        event_seq_.notify_all();
    }

    execute_team_member(0, team_size);

    // One branch-free pass over all flags lets their cache misses overlap.
    const auto joined = [&] {
        bool all = true;
        for (std::size_t tid = 1; tid < team_size; ++tid) all &= done_[tid].seq.load(std::memory_order_acquire) == seq;
        return all;
    };
    while (!spin_until_ready(joined)) std::this_thread::yield();

    auto error = std::move(team_error_);
    if (error) std::rethrow_exception(error);
}

void WorkerPool::run_ranges(std::size_t items, std::size_t min_grain, std::size_t scratch_bytes, void* context,
                            RangeFn fn) {
    if (items == 0) return;
    min_grain = std::max<std::size_t>(1, min_grain);
    const auto max_by_grain = (items + min_grain - 1) / min_grain;
    const auto team_size = std::min(max_by_grain, total_threads());

    if (team_size <= 1 || active_team_pool == this) {
        const InlineScratchScope inline_scratch(scratch_bytes);
        const WorkerContext worker{
            .tid = 0,
            .thread_count = 1,
            .scratch = inline_scratch.data(),
            .scratch_size = scratch_bytes,
        };
        fn(context, worker, 0, items);
        return;
    }

    std::scoped_lock caller_guard(caller_lock_);

    if (scratch_bytes > 0) {
        for (std::size_t t = 0; t < team_size; ++t) scratches_[t].ensure(scratch_bytes);
    }

    const auto total_units = (items + min_grain - 1) / min_grain;
    const auto base_units = total_units / team_size;
    const auto rem_units = total_units % team_size;
    std::size_t cursor = 0;
    for (std::size_t t = 0; t < team_size; ++t) {
        const auto units = base_units + (t < rem_units ? 1 : 0);
        const auto end = std::min(items, cursor + units * min_grain);
        const auto len = end - cursor;
        const auto home_units = std::max<std::size_t>(1, (units + 3) / 4);
        range_slots_[t].next.store(cursor, std::memory_order_relaxed);
        range_slots_[t].end = end;
        range_slots_[t].home_grain = std::max(min_grain, std::min(len, home_units * min_grain));
        range_slots_[t].steal_grain = min_grain;
        cursor = end;
    }

    team_.range = {range_slots_.get(), context, fn};

    const TeamFn runner = [](void* raw, const WorkerContext& worker, PhaseBarrier& barrier) {
        const auto p = *static_cast<const RangePayload*>(raw);
        auto& home = p.slots[worker.tid];
        for (;;) {
            if (barrier.aborted()) return;
            const auto cur = home.next.fetch_add(home.home_grain, std::memory_order_relaxed);
            if (cur >= home.end) break;
            p.user_fn(p.user_ctx, worker, cur, std::min(home.end, cur + home.home_grain));
        }
        for (std::size_t offset = 1; offset < worker.thread_count; ++offset) {
            if (barrier.aborted()) return;
            auto& victim = p.slots[(worker.tid + offset) % worker.thread_count];
            if (victim.next.load(std::memory_order_relaxed) >= victim.end) continue;
            for (;;) {
                if (barrier.aborted()) return;
                const auto cur = victim.next.fetch_add(victim.steal_grain, std::memory_order_relaxed);
                if (cur >= victim.end) break;
                p.user_fn(p.user_ctx, worker, cur, std::min(victim.end, cur + victim.steal_grain));
            }
        }
    };

    // Only an exception aborts the barrier; ranges never wait on it, so skip the reset (and its cache-line writes)
    // otherwise.
    if (team_barrier_.aborted()) team_barrier_.reset(team_size);
    dispatch(team_size, runner, &team_.range, scratch_bytes);
}

void WorkerPool::run_tasks(std::size_t n, void* context, TaskFn fn) {
    if (n == 0) return;
    struct Payload {
        void* ctx;
        TaskFn fn;
    } payload{context, fn};
    run_ranges(n, 1, 0, &payload, [](void* raw, const WorkerContext&, std::size_t begin, std::size_t end) {
        const auto p = *static_cast<const Payload*>(raw);
        for (std::size_t i = begin; i < end; ++i) p.fn(p.ctx, i);
    });
}

void WorkerPool::parallel_for(std::size_t n, const std::function<void(std::size_t)>& body) {
    if (n == 0) return;
    run_ranges(n, 1, 0, const_cast<std::function<void(std::size_t)>*>(&body),
               [](void* raw, const WorkerContext&, std::size_t begin, std::size_t end) {
                   const auto& fn = *static_cast<const std::function<void(std::size_t)>*>(raw);
                   for (std::size_t i = begin; i < end; ++i) fn(i);
               });
}

void WorkerPool::worker_main(std::size_t tid) {
#if defined(__APPLE__)
    if (::pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) != 0)
        (void)::pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
    std::uint64_t last_team_seq = 0;
    for (;;) {
        const auto seen = event_seq_.load(std::memory_order_acquire);
        if (stop_.load(std::memory_order_acquire)) break;

        const auto token = team_.token.load(std::memory_order_acquire);
        const auto seq = token >> 16;
        if (seq != last_team_seq) {
            last_team_seq = seq;
            const auto team_size = static_cast<std::size_t>(token & 0xFFFFu);
            if (tid < team_size) {
                execute_team_member(tid, team_size);
                done_[tid].seq.store(seq, std::memory_order_release);
                continue;
            }
        }

        idle_worker(seen, last_team_seq);
    }
}

void WorkerPool::idle_worker(std::uint64_t seen, std::uint64_t last_team_seq) {
    const auto fresh = [&] { return (team_.token.load(std::memory_order_acquire) >> 16) != last_team_seq; };
    if (spin_until_ready(fresh)) return;
    sleeping_.fetch_add(1, std::memory_order_seq_cst);
    if ((team_.token.load(std::memory_order_seq_cst) >> 16) == last_team_seq &&
        event_seq_.load(std::memory_order_seq_cst) == seen && !stop_.load(std::memory_order_acquire)) {
        event_seq_.wait(seen, std::memory_order_acquire);
    }
    sleeping_.fetch_sub(1, std::memory_order_relaxed);
}

void set_thread_count(std::size_t total_threads) noexcept {
    ensure_interactive_qos();
    configured_thread_count.store(total_threads, std::memory_order_relaxed);
}

std::size_t thread_count() noexcept {
    const auto configured = configured_thread_count.load(std::memory_order_relaxed);
    return configured == 0 ? default_thread_count() : configured;
}

ynn_status reserve_thread_pool(std::size_t total_threads) {
    if (total_threads == 0) total_threads = thread_count();
    if (total_threads <= 1) return ynn_status_success;
    return borrow_thread_pool(total_threads) ? ynn_status_success : ynn_status_invalid_parameter;
}

ynn_status parallel_for(std::size_t size, const std::function<void(std::size_t)>& body) {
    if (size == 0) return ynn_status_success;
    const auto threads = thread_count();
    if (size == 1 || threads <= 1) {
        for (std::size_t index = 0; index < size; ++index) body(index);
        return ynn_status_success;
    }
    auto* pool = borrow_thread_pool(threads);
    if (!pool) return ynn_status_invalid_parameter;
    pool->parallel_for(size, body);
    return ynn_status_success;
}

ynn_status parallel_tasks(std::size_t tasks, void* context, void (*body)(void*, std::size_t)) {
    if (tasks == 0) return ynn_status_success;
    const auto threads = thread_count();
    if (tasks == 1 || threads <= 1) {
        for (std::size_t index = 0; index < tasks; ++index) body(context, index);
        return ynn_status_success;
    }
    auto* pool = borrow_thread_pool(threads);
    if (!pool) return ynn_status_invalid_parameter;
    pool->run_tasks(tasks, context, body);
    return ynn_status_success;
}

ynn_status parallel_ranges(std::size_t items, std::size_t min_grain, void* context,
                           void (*body)(void*, std::size_t, std::size_t)) {
    struct Payload {
        void* ctx;
        void (*fn)(void*, std::size_t, std::size_t);
    } payload{context, body};
    return parallel_ranges_with_worker(items, min_grain, 0, &payload,
                                       [](void* raw, const WorkerContext&, std::size_t begin, std::size_t end) {
                                           const auto p = *static_cast<const Payload*>(raw);
                                           p.fn(p.ctx, begin, end);
                                       });
}

ynn_status parallel_ranges_with_worker(std::size_t items, std::size_t min_grain, std::size_t scratch_bytes,
                                       void* context,
                                       void (*body)(void*, const WorkerContext&, std::size_t, std::size_t)) {
    if (items == 0) return ynn_status_success;
    min_grain = std::max<std::size_t>(1, min_grain);
    const auto threads = thread_count();
    auto* pool = borrow_thread_pool(threads);
    if (!pool) return ynn_status_invalid_parameter;
    pool->run_ranges(items, min_grain, scratch_bytes, context, body);
    return ynn_status_success;
}

ynn_status parallel_team(std::size_t max_threads, std::size_t scratch_bytes, void* context,
                         void (*body)(void*, const WorkerContext&, PhaseBarrier&)) {
    if (max_threads == 0) return ynn_status_success;
    const auto threads = thread_count();
    const auto team = std::min(max_threads, threads);
    auto* pool = borrow_thread_pool(threads);
    if (!pool) return ynn_status_invalid_parameter;
    pool->run_team(team, scratch_bytes, context, body);
    return ynn_status_success;
}

std::uint64_t supported_arch_flags() noexcept { return get_supported_arch_flags(); }

std::string supported_arch_names() {
    const auto flags = supported_arch_flags();
    std::string result;
    const auto append = [&](std::uint64_t flag, const char* name) {
        if ((flags & flag) == 0) return;
        if (!result.empty()) result += ',';
        result += name;
    };
#ifdef YNN_ARCH_X86
    append(arch_flag::sse2, "sse2");
    append(arch_flag::ssse3, "ssse3");
    append(arch_flag::sse41, "sse41");
    append(arch_flag::avx, "avx");
    append(arch_flag::f16c, "f16c");
    append(arch_flag::avx2, "avx2");
    append(arch_flag::fma3, "fma3");
    append(arch_flag::avx512f, "avx512f");
    append(arch_flag::avx512bw, "avx512bw");
    append(arch_flag::avx512vl, "avx512vl");
    append(arch_flag::avx512dq, "avx512dq");
    append(arch_flag::avx512bf16, "avx512bf16");
    append(arch_flag::avx512fp16, "avx512fp16");
    append(arch_flag::avx512vnni, "avx512vnni");
    append(arch_flag::amxbf16, "amxbf16");
    append(arch_flag::amxfp16, "amxfp16");
    append(arch_flag::amxint8, "amxint8");
#endif
#ifdef YNN_ARCH_ARM
    append(arch_flag::neon, "neon");
    append(arch_flag::neonfma, "fma");
    append(arch_flag::neondot, "dotprod");
    append(arch_flag::neonfp16, "fp16");
    append(arch_flag::neonfp16arith, "fp16arith");
    append(arch_flag::neonbf16, "bf16");
    append(arch_flag::neonfp8, "fp8");
    append(arch_flag::neonfp8dot4, "fp8dot4");
    append(arch_flag::neoni8mm, "i8mm");
    append(arch_flag::sme, "sme");
    append(arch_flag::sme2, "sme2");
    append(arch_flag::sve, "sve");
#endif
    return result.empty() ? "generic" : result;
}

} // namespace ynn
