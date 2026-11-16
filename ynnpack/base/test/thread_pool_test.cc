// Licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.

#include "ynnpack/base/thread_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <thread>
#include <vector>

namespace ynn {
namespace {

constexpr int sum_below(int n) { return n * (n - 1) / 2; }

TEST(ThreadPoolTest, ParallelFor) {
    for (const int workers : {0, 1, 3, 7}) {
        WorkerPool pool(workers);
        EXPECT_EQ(pool.thread_count(), workers);
        EXPECT_EQ(pool.total_threads(), static_cast<size_t>(workers + 1));
        for (int n = 0; n < 100; ++n) {
            std::atomic<int> count = 0;
            std::atomic<int> sum = 0;
            pool.parallel_for(static_cast<size_t>(n), [&](size_t i) {
                ++count;
                sum += static_cast<int>(i);
            });
            EXPECT_EQ(count.load(), n);
            EXPECT_EQ(sum.load(), sum_below(n));
        }
    }
}

TEST(ThreadPoolTest, NestedParallelForCompletesInline) {
    WorkerPool pool(3);
    std::atomic<int> count = 0;
    std::atomic<int> sum = 0;
    pool.parallel_for(10, [&](size_t) {
        pool.parallel_for(8, [&](size_t j) {
            ++count;
            sum += static_cast<int>(j);
        });
    });
    EXPECT_EQ(count.load(), 80);
    EXPECT_EQ(sum.load(), 10 * sum_below(8));
}

TEST(ThreadPoolTest, RunTeamAndPhaseBarrier) {
    WorkerPool pool(3);
    std::vector<int> phase1(4, 0);
    std::vector<int> phase2(4, 0);
    struct State {
        std::vector<int>* p1;
        std::vector<int>* p2;
    } state{&phase1, &phase2};
    pool.run_team(4, 256, &state, [](void* raw, const WorkerContext& worker, PhaseBarrier& barrier) {
        auto* s = static_cast<State*>(raw);
        (*s->p1)[worker.tid] = static_cast<int>(worker.tid + 10);
        if (!barrier.arrive_and_wait()) return;
        int total = 0;
        for (int v : *s->p1) total += v;
        (*s->p2)[worker.tid] = total + static_cast<int>(worker.tid);
        (void)barrier.arrive_and_wait();
    });
    constexpr int kExpectedP1Sum = 10 + 11 + 12 + 13;
    for (int t = 0; t < 4; ++t) {
        EXPECT_EQ(phase2[t], kExpectedP1Sum + t);
    }

    for (int iter = 0; iter < 1000; ++iter) {
        const size_t team = (iter % 2 == 0) ? 2 : 4;
        std::atomic<size_t> seen = 0;
        pool.run_team(team, 0, &seen, [](void* raw, const WorkerContext&, PhaseBarrier& barrier) {
            static_cast<std::atomic<size_t>*>(raw)->fetch_add(1, std::memory_order_relaxed);
            (void)barrier.arrive_and_wait();
        });
        EXPECT_EQ(seen.load(std::memory_order_relaxed), team);
    }
}

TEST(ThreadPoolTest, RunRangesStealingAndScratch) {
    WorkerPool pool(3);
    constexpr size_t N = 256;
    std::vector<std::atomic<int>> hits(N);
    for (auto& h : hits) h.store(0, std::memory_order_relaxed);
    std::atomic<bool> aligned_ok = true;
    struct Ctx {
        std::vector<std::atomic<int>>* hits;
        std::atomic<bool>* aligned_ok;
    } ctx{&hits, &aligned_ok};
    pool.run_ranges(N, 2, 1024, &ctx, [](void* raw, const WorkerContext& worker, size_t begin, size_t end) {
        auto* c = static_cast<Ctx*>(raw);
        if (worker.scratch == nullptr || (reinterpret_cast<uintptr_t>(worker.scratch) % 64 != 0) ||
            worker.scratch_size < 1024) {
            c->aligned_ok->store(false, std::memory_order_relaxed);
        }
        auto* buf = reinterpret_cast<size_t*>(worker.scratch);
        buf[0] = worker.tid;
        if (worker.tid == 1 && begin < 80) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        if (buf[0] != worker.tid) c->aligned_ok->store(false, std::memory_order_relaxed);
        for (size_t i = begin; i < end; ++i) c->hits->at(i).fetch_add(1, std::memory_order_relaxed);
    });
    EXPECT_TRUE(aligned_ok.load());
    for (const auto& h : hits) EXPECT_EQ(h.load(std::memory_order_relaxed), 1);
}

TEST(ThreadPoolTest, ParallelRangeSlotsDrain) {
    WorkerPool pool(3);
    constexpr size_t kItems = 97;
    std::vector<std::atomic<int>> hits(kItems);
    for (auto& h : hits) h.store(0, std::memory_order_relaxed);
    ParallelRangeSlots slots(kItems, 4);
    struct State {
        ParallelRangeSlots* slots;
        std::vector<std::atomic<int>>* hits;
    } state{&slots, &hits};
    pool.run_team(4, 0, &state, [](void* raw, const WorkerContext& worker, PhaseBarrier& barrier) {
        auto* s = static_cast<State*>(raw);
        drain_range_slots(*s->slots, worker.tid, 3, [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) s->hits->at(i).fetch_add(1, std::memory_order_relaxed);
        });
        (void)barrier.arrive_and_wait();
    });
    for (const auto& h : hits) EXPECT_EQ(h.load(std::memory_order_relaxed), 1);
}

TEST(ThreadPoolTest, NestedInlineScratch) {
    for (const int workers : {0, 3}) {
        WorkerPool pool(workers);
        std::atomic<bool> intact = true;
        struct State {
            WorkerPool* pool;
            std::atomic<bool>* intact;
        } state{&pool, &intact};
        pool.run_ranges(8, 1, 128, &state, [](void* raw, const WorkerContext& outer, size_t begin, size_t) {
            auto* s = static_cast<State*>(raw);
            if (outer.scratch == nullptr || outer.scratch_size < 128) {
                s->intact->store(false, std::memory_order_relaxed);
                return;
            }
            for (size_t i = 0; i < 128; ++i) outer.scratch[i] = static_cast<std::byte>((begin + i) & 0xFFu);

            s->pool->run_ranges(4, 1, 65536, s, [](void* inner_raw, const WorkerContext& mid, size_t, size_t) {
                auto* is = static_cast<State*>(inner_raw);
                if (mid.scratch == nullptr || mid.scratch_size < 65536) {
                    is->intact->store(false, std::memory_order_relaxed);
                    return;
                }
                for (size_t k = 0; k < 65536; ++k) mid.scratch[k] = std::byte{0xA5};
                is->pool->run_team(2, 131072, is, [](void* leaf_raw, const WorkerContext& leaf, PhaseBarrier& b) {
                    auto* ls = static_cast<State*>(leaf_raw);
                    if (leaf.scratch == nullptr || leaf.scratch_size < 131072) {
                        ls->intact->store(false, std::memory_order_relaxed);
                        return;
                    }
                    for (size_t m = 0; m < 131072; ++m) leaf.scratch[m] = std::byte{0x5A};
                    (void)b.arrive_and_wait();
                });
                for (size_t k = 0; k < 65536; ++k) {
                    if (mid.scratch[k] != std::byte{0xA5}) {
                        is->intact->store(false, std::memory_order_relaxed);
                        break;
                    }
                }
            });

            for (size_t i = 0; i < 128; ++i) {
                if (outer.scratch[i] != static_cast<std::byte>((begin + i) & 0xFFu)) {
                    s->intact->store(false, std::memory_order_relaxed);
                    break;
                }
            }
        });
        EXPECT_TRUE(intact.load());
    }
}

TEST(ThreadPoolTest, ExceptionPropagationAndGlobalHelpers) {
    WorkerPool pool(3);
    bool caught = false;
    try {
        pool.parallel_for(32, [](size_t index) {
            if (index == 17) throw 42;
        });
    } catch (int value) {
        caught = (value == 42);
    }
    EXPECT_TRUE(caught);

    bool barrier_caught = false;
    try {
        pool.run_team(4, 0, nullptr, [](void*, const WorkerContext& worker, PhaseBarrier& barrier) {
            if (worker.tid == 2) throw 99;
            (void)barrier.arrive_and_wait();
        });
    } catch (int value) {
        barrier_caught = (value == 99);
    }
    EXPECT_TRUE(barrier_caught);

    set_thread_count(2);
    EXPECT_EQ(thread_count(), 2u);
    EXPECT_EQ(reserve_thread_pool(2), ynn_status_success);
    std::atomic<int> sum = 0;
    EXPECT_EQ(parallel_ranges(16, 1,
                              [&](size_t begin, size_t end) {
                                  for (size_t i = begin; i < end; ++i)
                                      sum.fetch_add(static_cast<int>(i), std::memory_order_relaxed);
                              }),
              ynn_status_success);
    EXPECT_EQ(sum.load(), sum_below(16));
    set_thread_count(0);
    EXPECT_FALSE(supported_arch_names().empty());
}

} // namespace
} // namespace ynn
