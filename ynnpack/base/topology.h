// Licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_TOPOLOGY_H_
#define XNNPACK_YNNPACK_BASE_TOPOLOGY_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ynn {

struct NumaNode {
    int id = 0;
    std::vector<int> cpus;
};

struct CpuCapacity {
    int cpu = 0;
    std::uint64_t score = 0;
};

std::vector<int> parse_cpu_list(std::string_view text);
std::optional<std::size_t> choose_numa_node(std::span<const NumaNode> nodes, int current_cpu, std::size_t threads);
std::size_t count_performance_cpus(std::span<const CpuCapacity> cpus);
std::size_t default_thread_count() noexcept;
std::vector<int> choose_performance_cpus(std::span<const CpuCapacity> cpus, std::size_t threads);
bool confine_to_performance_cpus(std::size_t threads);
std::optional<int> confine_to_numa_node(std::size_t threads);

} // namespace ynn

#endif // XNNPACK_YNNPACK_BASE_TOPOLOGY_H_
