// Licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.

#include "ynnpack/base/topology.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <thread>
#include <vector>

#if defined(__linux__) && !defined(__ANDROID__)
#include <sched.h>
#endif

namespace ynn {
namespace {

TEST(TopologyTest, ParseCpuList) {
    EXPECT_EQ(parse_cpu_list("0-3,8,10-11\n"), (std::vector<int>{0, 1, 2, 3, 8, 10, 11}));
    EXPECT_EQ(parse_cpu_list("5"), (std::vector<int>{5}));
    EXPECT_TRUE(parse_cpu_list("").empty());
    EXPECT_TRUE(parse_cpu_list("3-1").empty());
    EXPECT_TRUE(parse_cpu_list("a-b").empty());
    EXPECT_TRUE(parse_cpu_list("0-").empty());
    EXPECT_TRUE(parse_cpu_list("0,,2").empty());
    EXPECT_TRUE(parse_cpu_list("0-70000").empty());
}

TEST(TopologyTest, ChooseNumaNode) {
    const std::array<NumaNode, 2> two = {NumaNode{0, {0, 1, 2, 3}}, NumaNode{1, {4, 5, 6, 7}}};
    const std::array<NumaNode, 2> uneven = {NumaNode{0, {0, 1}}, NumaNode{1, {4, 5, 6, 7}}};
    const std::array<NumaNode, 1> one = {NumaNode{0, {0, 1, 2, 3}}};
    EXPECT_EQ(choose_numa_node(two, 5, 4), std::optional<size_t>(1));
    EXPECT_EQ(choose_numa_node(two, 1, 4), std::optional<size_t>(0));
    EXPECT_EQ(choose_numa_node(two, -1, 2), std::optional<size_t>(0));
    EXPECT_EQ(choose_numa_node(uneven, 0, 4), std::optional<size_t>(1));
    EXPECT_FALSE(choose_numa_node(two, 0, 5).has_value());
    EXPECT_FALSE(choose_numa_node(one, 0, 2).has_value());
    EXPECT_FALSE(choose_numa_node(two, 0, 0).has_value());
    EXPECT_FALSE(choose_numa_node({}, 0, 1).has_value());
}

TEST(TopologyTest, ChoosePerformanceCpus) {
    const std::array<CpuCapacity, 8> hybrid = {
        CpuCapacity{0, 448},  CpuCapacity{1, 448},  CpuCapacity{2, 448},  CpuCapacity{3, 448},
        CpuCapacity{4, 1024}, CpuCapacity{5, 1024}, CpuCapacity{6, 1024}, CpuCapacity{7, 1024},
    };
    const std::array<CpuCapacity, 8> tri_cluster = {
        CpuCapacity{0, 320}, CpuCapacity{1, 320}, CpuCapacity{2, 320}, CpuCapacity{3, 320},
        CpuCapacity{4, 768}, CpuCapacity{5, 768}, CpuCapacity{6, 768}, CpuCapacity{7, 1024},
    };
    const std::array<CpuCapacity, 4> uniform = {
        CpuCapacity{0, 1024},
        CpuCapacity{1, 1024},
        CpuCapacity{2, 1024},
        CpuCapacity{3, 1024},
    };
    EXPECT_EQ(choose_performance_cpus(hybrid, 4), (std::vector<int>{4, 5, 6, 7}));
    EXPECT_EQ(choose_performance_cpus(hybrid, 2), (std::vector<int>{4, 5, 6, 7}));
    EXPECT_EQ(choose_performance_cpus(tri_cluster, 4), (std::vector<int>{4, 5, 6, 7}));
    EXPECT_TRUE(choose_performance_cpus(hybrid, 8).empty());
    EXPECT_TRUE(choose_performance_cpus(uniform, 2).empty());
    EXPECT_TRUE(choose_performance_cpus(hybrid, 0).empty());
    EXPECT_EQ(count_performance_cpus(hybrid), 4u);
    EXPECT_EQ(count_performance_cpus(tri_cluster), 4u);
    EXPECT_EQ(count_performance_cpus(uniform), 4u);
    EXPECT_EQ(count_performance_cpus({}), 0u);
    EXPECT_GE(default_thread_count(), 1u);
}

TEST(TopologyTest, ConfineToNumaNode) {
#if defined(__linux__) && !defined(__ANDROID__)
    const auto affinity = [] {
        cpu_set_t set;
        CPU_ZERO(&set);
        std::vector<int> cpus;
        if (sched_getaffinity(0, sizeof(set), &set) != 0) return cpus;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &set)) cpus.push_back(cpu);
        return cpus;
    };
    std::thread([&] {
        const auto before = affinity();
        setenv("YNN_NUMA", "off", 1);
        EXPECT_FALSE(confine_to_numa_node(1).has_value());
        EXPECT_EQ(affinity(), before);
        unsetenv("YNN_NUMA");
        const auto node = confine_to_numa_node(1);
        const auto after = affinity();
        EXPECT_FALSE(after.empty());
        EXPECT_TRUE(std::ranges::includes(before, after));
        if (node) {
            EXPECT_LE(after.size(), before.size());
        } else {
            EXPECT_EQ(after, before);
        }
    }).join();
#else
    EXPECT_FALSE(confine_to_numa_node(1).has_value());
    EXPECT_FALSE(confine_to_performance_cpus(1));
#endif
}

} // namespace
} // namespace ynn
