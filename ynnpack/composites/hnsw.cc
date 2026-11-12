// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/composites/hnsw.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "slinky/base/thread_pool.h"

namespace ynn {
namespace {

struct ByDistance {
  bool operator()(const std::pair<float, uint32_t>& left,
                  const std::pair<float, uint32_t>& right) const noexcept {
    return left.first < right.first;
  }
};

inline void heap_push(std::vector<std::pair<float, uint32_t>>& heap,
                      float distance, uint32_t node) {
  heap.emplace_back(distance, node);
  std::push_heap(heap.begin(), heap.end(), ByDistance{});
}

inline std::pair<float, uint32_t> heap_pop(
    std::vector<std::pair<float, uint32_t>>& heap) {
  std::pop_heap(heap.begin(), heap.end(), ByDistance{});
  const auto result = heap.back();
  heap.pop_back();
  return result;
}

inline void graph_neighbors(const hnsw_graph& graph, uint32_t node,
                            size_t level, const uint32_t** out_ptr,
                            size_t* out_len) noexcept {
  if (level == 0) {
    const uint32_t start = graph.base_offsets[node];
    const uint32_t end = graph.base_offsets[node + 1];
    *out_ptr = graph.base_edges.data() + start;
    *out_len = static_cast<size_t>(end - start);
    return;
  }
  const hnsw_layer& layer = graph.upper[level - 1];
  const auto found = std::lower_bound(layer.nodes.begin(), layer.nodes.end(), node);
  if (found == layer.nodes.end() || *found != node) {
    *out_ptr = nullptr;
    *out_len = 0;
    return;
  }
  const auto idx = static_cast<size_t>(found - layer.nodes.begin());
  const uint32_t start = layer.offsets[idx];
  const uint32_t end = layer.offsets[idx + 1];
  *out_ptr = layer.edges.data() + start;
  *out_len = static_cast<size_t>(end - start);
}

hnsw_parallel_for_fn make_threadpool_parallel_for(ynn_threadpool_t threadpool) {
  if (threadpool == nullptr) return {};
  auto* pool = reinterpret_cast<slinky::thread_pool*>(threadpool);
  if (pool->thread_count() <= 1) return {};
  return [pool](size_t task_count, std::function<void(size_t)> body) -> bool {
    pool->parallel_for(task_count, [&body](size_t idx) { body(idx); });
    return true;
  };
}

void build_reordered_graph(const hnsw_graph& src, hnsw_graph* dst) {
  const size_t nodes = src.labels.size();
  std::vector<uint32_t> old_to_new(nodes,
                                   std::numeric_limits<uint32_t>::max());
  std::vector<uint32_t> new_to_old;
  new_to_old.reserve(nodes);
  const auto visit = [&](uint32_t u) {
    if (old_to_new[u] == std::numeric_limits<uint32_t>::max()) {
      old_to_new[u] = static_cast<uint32_t>(new_to_old.size());
      new_to_old.push_back(u);
    }
  };
  visit(src.entry);
  for (size_t level = src.upper.size(); level != 0; --level) {
    const hnsw_layer& layer = src.upper[level - 1];
    for (uint32_t u : layer.nodes) {
      visit(u);
      const uint32_t* edges = nullptr;
      size_t count = 0;
      graph_neighbors(src, u, level, &edges, &count);
      for (size_t i = 0; i < count; ++i) visit(edges[i]);
    }
  }
  size_t cursor = 0;
  for (uint32_t seed = 0; seed < nodes; ++seed) {
    visit(seed);
    while (cursor < new_to_old.size()) {
      const uint32_t u = new_to_old[cursor++];
      const uint32_t* edges = nullptr;
      size_t count = 0;
      graph_neighbors(src, u, 0, &edges, &count);
      for (size_t i = 0; i < count; ++i) visit(edges[i]);
    }
  }

  *dst = {};
  dst->width = src.width;
  dst->m = src.m;
  dst->entry = old_to_new[src.entry];
  dst->labels.resize(nodes);
  dst->levels.resize(nodes);
  dst->base_offsets.reserve(nodes + 1);
  dst->base_offsets.push_back(0);
  dst->base_edges.reserve(src.base_edges.size());
  for (size_t new_u = 0; new_u < nodes; ++new_u) {
    const uint32_t old_u = new_to_old[new_u];
    dst->labels[new_u] = src.labels[old_u];
    dst->levels[new_u] = src.levels[old_u];
    const uint32_t* edges = nullptr;
    size_t count = 0;
    graph_neighbors(src, old_u, 0, &edges, &count);
    for (size_t i = 0; i < count; ++i) {
      dst->base_edges.push_back(old_to_new[edges[i]]);
    }
    dst->base_offsets.push_back(static_cast<uint32_t>(dst->base_edges.size()));
  }

  dst->upper.resize(src.upper.size());
  for (size_t level = 1; level <= src.upper.size(); ++level) {
    const hnsw_layer& src_layer = src.upper[level - 1];
    hnsw_layer& dst_layer = dst->upper[level - 1];
    std::vector<std::pair<uint32_t, size_t>> active;
    active.reserve(src_layer.nodes.size());
    for (size_t idx = 0; idx < src_layer.nodes.size(); ++idx) {
      active.emplace_back(old_to_new[src_layer.nodes[idx]], idx);
    }
    std::sort(active.begin(), active.end());
    dst_layer.nodes.reserve(active.size());
    dst_layer.offsets.reserve(active.size() + 1);
    dst_layer.offsets.push_back(0);
    dst_layer.edges.reserve(src_layer.edges.size());
    for (const auto& item : active) {
      dst_layer.nodes.push_back(item.first);
      const size_t start = src_layer.offsets[item.second];
      const size_t end = src_layer.offsets[item.second + 1];
      for (size_t e = start; e < end; ++e) {
        dst_layer.edges.push_back(old_to_new[src_layer.edges[e]]);
      }
      dst_layer.offsets.push_back(static_cast<uint32_t>(dst_layer.edges.size()));
    }
  }
}

}  // namespace

size_t hnsw_graph::runtime_bytes() const noexcept {
  size_t bytes =
      (labels.size() + base_offsets.size() + base_edges.size()) *
          sizeof(uint32_t) +
      levels.size();
  for (const auto& layer : upper) {
    bytes +=
        (layer.nodes.size() + layer.offsets.size() + layer.edges.size()) *
        sizeof(uint32_t);
  }
  return bytes;
}

bool validate_hnsw_graph(const hnsw_graph& graph) {
  const size_t nodes = graph.labels.size();
  if (nodes == 0 || nodes > kHnswMaxNodes || graph.width == 0 ||
      graph.m < 2 || graph.m > 65535 || graph.entry >= nodes ||
      graph.levels.size() != nodes || graph.base_offsets.size() != nodes + 1 ||
      graph.base_offsets.front() != 0 ||
      graph.base_offsets.back() != graph.base_edges.size() ||
      graph.base_edges.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  std::vector<bool> seen_labels(nodes, false);
  for (uint32_t label : graph.labels) {
    if (label >= nodes || seen_labels[label]) return false;
    seen_labels[label] = true;
  }
  for (size_t node = 0; node < nodes; ++node) {
    const uint32_t start = graph.base_offsets[node];
    const uint32_t end = graph.base_offsets[node + 1];
    if (start > end || end > graph.base_edges.size() ||
        end - start > 2 * graph.m) {
      return false;
    }
  }
  for (uint32_t edge : graph.base_edges) {
    if (edge >= nodes) return false;
  }
  std::vector<uint8_t> derived_levels(nodes, 0);
  for (size_t level = 1; level <= graph.upper.size(); ++level) {
    const hnsw_layer& layer = graph.upper[level - 1];
    if (layer.nodes.empty() || layer.offsets.size() != layer.nodes.size() + 1 ||
        layer.offsets.front() != 0 ||
        layer.offsets.back() != layer.edges.size() ||
        layer.edges.size() > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    for (size_t idx = 0; idx < layer.nodes.size(); ++idx) {
      const uint32_t node = layer.nodes[idx];
      if (node >= nodes || (idx != 0 && node <= layer.nodes[idx - 1]) ||
          derived_levels[node] != level - 1 ||
          layer.offsets[idx] > layer.offsets[idx + 1] ||
          layer.offsets[idx + 1] > layer.edges.size() ||
          layer.offsets[idx + 1] - layer.offsets[idx] > graph.m) {
        return false;
      }
      derived_levels[node] = static_cast<uint8_t>(level);
    }
    for (uint32_t edge : layer.edges) {
      if (edge >= nodes || derived_levels[edge] != level) return false;
    }
  }
  return derived_levels == graph.levels &&
         graph.levels[graph.entry] == graph.upper.size();
}

ynn_status build_hnsw_graph(const void* vectors, bool vectors_bf16,
                            size_t num_nodes, size_t width,
                            const hnsw_build_options& options,
                            hnsw_graph* out_graph,
                            hnsw_progress_fn progress,
                            hnsw_parallel_for_fn parallel_for) {
  if (vectors == nullptr || out_graph == nullptr || num_nodes == 0 ||
      num_nodes > kHnswMaxNodes || width == 0 ||
      width > std::numeric_limits<uint32_t>::max() || options.m < 2 ||
      options.m > 65535 || options.ef_construction < options.m) {
    return ynn_status_invalid_parameter;
  }

  const auto* table = static_cast<const uint8_t*>(vectors);
  const size_t row_bytes = width * (vectors_bf16 ? sizeof(uint16_t) : sizeof(float));
  const hnsw_dot_kernels kernels = get_hnsw_dot_kernels(vectors_bf16);

  hnsw_graph graph;
  graph.width = static_cast<uint32_t>(width);
  graph.m = options.m;
  graph.labels.resize(num_nodes);
  std::iota(graph.labels.begin(), graph.labels.end(), 0u);
  graph.levels.resize(num_nodes);

  std::mt19937_64 level_rng(options.seed);
  std::uniform_real_distribution<double> uniform(std::nextafter(0.0, 1.0), 1.0);
  const double multiplier = 1.0 / std::log(static_cast<double>(options.m));
  size_t maximum_level = 0;
  size_t total_slots = 0;
  for (size_t node = 0; node < num_nodes; ++node) {
    const auto sampled =
        static_cast<size_t>(-std::log(uniform(level_rng)) * multiplier);
    const auto level = static_cast<uint8_t>(std::min<size_t>(sampled, 255));
    graph.levels[node] = level;
    maximum_level = std::max<size_t>(maximum_level, level);
    total_slots += static_cast<size_t>(level) + 1;
  }

  std::vector<std::vector<uint32_t>> level_buckets(maximum_level + 1);
  for (size_t node = 0; node < num_nodes; ++node) {
    level_buckets[graph.levels[node]].push_back(static_cast<uint32_t>(node));
  }
  std::mt19937_64 perm_rng(options.seed ^ 0x9E3779B97F4A7C15ULL);
  for (auto& bucket : level_buckets) {
    for (size_t i = bucket.size(); i > 1; --i) {
      std::swap(bucket[i - 1], bucket[static_cast<size_t>(perm_rng() % i)]);
    }
  }

  std::vector<size_t> node_slot(num_nodes);
  std::vector<size_t> slot_offset(total_slots + 1, 0);
  std::vector<uint32_t> slot_size(total_slots, 0);
  size_t slot_cursor = 0;
  size_t edge_cursor = 0;
  for (size_t node = 0; node < num_nodes; ++node) {
    node_slot[node] = slot_cursor;
    const size_t top = graph.levels[node];
    for (size_t lvl = 0; lvl <= top; ++lvl) {
      slot_offset[slot_cursor++] = edge_cursor;
      edge_cursor += (lvl == 0 ? 2 : 1) * static_cast<size_t>(options.m);
    }
  }
  slot_offset[total_slots] = edge_cursor;
  std::vector<uint32_t> slot_edges(edge_cursor);

  std::atomic<bool> nonfinite{false};
  const auto row_ptr = [&](uint32_t node) noexcept -> const uint8_t* {
    return table + static_cast<size_t>(node) * row_bytes;
  };
  const auto row_distance = [&](uint32_t left, uint32_t right) noexcept -> float {
    const float result = 1.0f - kernels.row_dot(row_ptr(left), row_ptr(right), width);
    if (!std::isfinite(result)) {
      nonfinite.store(true, std::memory_order_relaxed);
      return 0.0f;
    }
    return result;
  };
  const auto score_nodes = [&](uint32_t query, const uint32_t* ids,
                               size_t count, std::vector<float>& out) noexcept {
    out.resize(count);
    const uint8_t* q = row_ptr(query);
    size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
      if (idx + 8 <= count) {
        hnsw_prefetch_slice(row_ptr(ids[idx + 4]), row_bytes);
        hnsw_prefetch_slice(row_ptr(ids[idx + 5]), row_bytes);
        hnsw_prefetch_slice(row_ptr(ids[idx + 6]), row_bytes);
        hnsw_prefetch_slice(row_ptr(ids[idx + 7]), row_bytes);
      }
      float dots[4];
      kernels.row_dot4(q, row_ptr(ids[idx + 0]), row_ptr(ids[idx + 1]),
                       row_ptr(ids[idx + 2]), row_ptr(ids[idx + 3]), width,
                       dots);
      for (size_t r = 0; r < 4; ++r) {
        const float d = 1.0f - dots[r];
        if (!std::isfinite(d)) {
          nonfinite.store(true, std::memory_order_relaxed);
          out[idx + r] = 0.0f;
        } else {
          out[idx + r] = d;
        }
      }
    }
    for (; idx < count; ++idx) {
      out[idx] = row_distance(query, ids[idx]);
    }
  };
  const auto slot_range = [&](uint32_t u, size_t lvl, const uint32_t** ptr,
                              size_t* len) noexcept {
    const size_t s = node_slot[u] + lvl;
    *ptr = slot_edges.data() + slot_offset[s];
    *len = slot_size[s];
  };
  const auto write_neighbors = [&](uint32_t u, size_t lvl,
                                   const std::vector<uint32_t>& list) noexcept {
    const size_t s = node_slot[u] + lvl;
    slot_size[s] = static_cast<uint32_t>(list.size());
    if (!list.empty()) {
      std::memcpy(slot_edges.data() + slot_offset[s], list.data(),
                  list.size() * sizeof(uint32_t));
    }
  };

  using Pair = std::pair<float, uint32_t>;
  struct Workspace {
    std::vector<uint8_t> visited;
    uint8_t epoch = 0;
    std::vector<Pair> nearest, candidates, scored;
    std::vector<uint32_t> unseen, selected, rejected, merged;
    std::vector<float> distances;
  };
  std::array<Workspace, 64> workspaces;

  const auto select_neighbors = [&](const std::vector<Pair>& sorted_pool,
                                    size_t limit, std::vector<uint32_t>& out,
                                    std::vector<uint32_t>& rejected) {
    out.clear();
    out.reserve(std::min(sorted_pool.size(), limit));
    if (sorted_pool.size() <= limit) {
      for (const auto& item : sorted_pool) out.push_back(item.second);
      return;
    }
    rejected.clear();
    for (const auto& item : sorted_pool) {
      const float dist = item.first;
      const uint32_t cand = item.second;
      bool keep = true;
      for (uint32_t kept : out) {
        if (row_distance(cand, kept) < dist) {
          keep = false;
          break;
        }
      }
      if (keep) {
        out.push_back(cand);
        if (out.size() == limit) return;
      } else if (out.size() + rejected.size() < limit) {
        rejected.push_back(cand);
      }
    }
    for (uint32_t cand : rejected) {
      if (out.size() == limit) break;
      out.push_back(cand);
    }
  };

  const auto search_layer = [&](Workspace& ws, uint32_t query, uint32_t entry,
                                float entry_dist, size_t ef, size_t level) {
    if (ws.visited.empty()) ws.visited.assign(num_nodes, 0);
    if (++ws.epoch == 0) {
      std::fill(ws.visited.begin(), ws.visited.end(), uint8_t{0});
      ws.epoch = 1;
    }
    ws.nearest.clear();
    ws.candidates.clear();
    ws.nearest.emplace_back(entry_dist, entry);
    ws.candidates.emplace_back(entry_dist, entry);
    ws.visited[entry] = ws.epoch;
    bool nearest_heap = ef == 1;
    while (!ws.candidates.empty()) {
      if (nearest_heap &&
          ws.candidates.front().first > ws.nearest.front().first) {
        break;
      }
      std::pop_heap(ws.candidates.begin(), ws.candidates.end(),
                    std::greater<Pair>{});
      const uint32_t current = ws.candidates.back().second;
      ws.candidates.pop_back();
      ws.unseen.clear();
      const uint32_t* nbs = nullptr;
      size_t nbs_len = 0;
      slot_range(current, level, &nbs, &nbs_len);
      for (size_t i = 0; i < nbs_len; ++i) {
        const uint32_t nb = nbs[i];
        if (ws.visited[nb] == ws.epoch) continue;
        ws.visited[nb] = ws.epoch;
        ws.unseen.push_back(nb);
      }
      if (ws.unseen.empty()) continue;
      score_nodes(query, ws.unseen.data(), ws.unseen.size(), ws.distances);
      for (size_t idx = 0; idx < ws.unseen.size(); ++idx) {
        const float d = ws.distances[idx];
        const uint32_t nb = ws.unseen[idx];
        if (!nearest_heap) {
          ws.nearest.emplace_back(d, nb);
          if (ws.nearest.size() == ef) {
            std::make_heap(ws.nearest.begin(), ws.nearest.end());
            nearest_heap = true;
          }
          ws.candidates.emplace_back(d, nb);
          std::push_heap(ws.candidates.begin(), ws.candidates.end(),
                         std::greater<Pair>{});
        } else if (d < ws.nearest.front().first) {
          std::pop_heap(ws.nearest.begin(), ws.nearest.end());
          ws.nearest.back() = {d, nb};
          std::push_heap(ws.nearest.begin(), ws.nearest.end());
          ws.candidates.emplace_back(d, nb);
          std::push_heap(ws.candidates.begin(), ws.candidates.end(),
                         std::greater<Pair>{});
        }
      }
    }
    std::sort(ws.nearest.begin(), ws.nearest.end());
  };

  const auto run_tasks = [&](size_t items, size_t min_parallel,
                             auto&& worker) -> bool {
    if (items == 0) return true;
    if (!parallel_for || items < min_parallel) {
      worker(0, 0, items);
      return !nonfinite.load(std::memory_order_relaxed);
    }
    const size_t tasks = std::min<size_t>(workspaces.size(), (items + 1) / 2);
    const size_t chunk = (items + tasks - 1) / tasks;
    if (!parallel_for(tasks, [&](size_t task) {
          const size_t begin = task * chunk;
          if (begin >= items || nonfinite.load(std::memory_order_relaxed)) {
            return;
          }
          worker(task, begin, std::min(items, begin + chunk));
        })) {
      return false;
    }
    return !nonfinite.load(std::memory_order_relaxed);
  };

  graph.entry = level_buckets[maximum_level].front();
  (void)row_distance(graph.entry, graph.entry);
  if (nonfinite.load(std::memory_order_relaxed)) return ynn_status_error;
  size_t completed = 1;
  if (progress) progress(1, num_nodes);

  const size_t max_batch = std::max<size_t>(16, num_nodes / 50);
  struct ReverseEdge {
    uint32_t dest;
    uint32_t src;
    uint8_t level;
  };
  std::vector<ReverseEdge> reverse_edges;
  std::array<size_t, 256> bucket_counts{}, bucket_offsets{};
  std::vector<uint32_t> active_buckets;
  active_buckets.reserve(256);

  for (size_t bucket_level = maximum_level + 1; bucket_level-- > 0;) {
    const auto& bucket = level_buckets[bucket_level];
    size_t cursor = bucket_level == maximum_level ? 1 : 0;
    while (cursor < bucket.size()) {
      const size_t step =
          std::min({bucket.size() - cursor, completed, max_batch});
      const uint32_t* batch = bucket.data() + cursor;
      cursor += step;

      if (!run_tasks(step, 8, [&](size_t task, size_t begin, size_t end) {
            Workspace& ws = workspaces[task];
            for (size_t item = begin; item < end; ++item) {
              const uint32_t u = batch[item];
              const size_t pt_level = graph.levels[u];
              uint32_t curr = graph.entry;
              float curr_dist = row_distance(u, curr);
              for (size_t lvl = maximum_level; lvl > pt_level; --lvl) {
                bool changed = true;
                while (changed) {
                  changed = false;
                  const uint32_t* nbs = nullptr;
                  size_t nbs_len = 0;
                  slot_range(curr, lvl, &nbs, &nbs_len);
                  score_nodes(u, nbs, nbs_len, ws.distances);
                  for (size_t idx = 0; idx < nbs_len; ++idx) {
                    if (ws.distances[idx] < curr_dist) {
                      curr_dist = ws.distances[idx];
                      curr = nbs[idx];
                      changed = true;
                    }
                  }
                }
              }
              for (size_t offset = 0; offset <= pt_level; ++offset) {
                const size_t lvl = pt_level - offset;
                search_layer(ws, u, curr, curr_dist, options.ef_construction,
                             lvl);
                if (!ws.nearest.empty() &&
                    ws.nearest.front().first < curr_dist) {
                  curr_dist = ws.nearest.front().first;
                  curr = ws.nearest.front().second;
                }
                select_neighbors(ws.nearest, options.m, ws.selected,
                                 ws.rejected);
                write_neighbors(u, lvl, ws.selected);
              }
            }
          })) {
        return ynn_status_error;
      }

      bucket_counts.fill(0);
      for (size_t item = 0; item < step; ++item) {
        const uint32_t u = batch[item];
        const size_t top = graph.levels[u];
        for (size_t lvl = 0; lvl <= top; ++lvl) {
          const uint32_t* nbs = nullptr;
          size_t nbs_len = 0;
          slot_range(u, lvl, &nbs, &nbs_len);
          for (size_t i = 0; i < nbs_len; ++i) ++bucket_counts[nbs[i] & 255u];
        }
      }
      size_t total_edges = 0;
      active_buckets.clear();
      for (size_t b = 0; b < 256; ++b) {
        bucket_offsets[b] = total_edges;
        if (bucket_counts[b] != 0) {
          active_buckets.push_back(static_cast<uint32_t>(b));
        }
        total_edges += bucket_counts[b];
      }
      reverse_edges.resize(total_edges);
      auto write_pos = bucket_offsets;
      for (size_t item = 0; item < step; ++item) {
        const uint32_t u = batch[item];
        const size_t top = graph.levels[u];
        for (size_t lvl = 0; lvl <= top; ++lvl) {
          const uint32_t* nbs = nullptr;
          size_t nbs_len = 0;
          slot_range(u, lvl, &nbs, &nbs_len);
          for (size_t i = 0; i < nbs_len; ++i) {
            const uint32_t dest = nbs[i];
            reverse_edges[write_pos[dest & 255u]++] = {
                dest, u, static_cast<uint8_t>(lvl)};
          }
        }
      }

      const size_t min_buckets =
          total_edges >= 64 ? 2 : active_buckets.size() + 1;
      if (!run_tasks(
              active_buckets.size(), min_buckets,
              [&](size_t task, size_t begin, size_t end) {
                Workspace& ws = workspaces[task];
                for (size_t idx = begin; idx < end; ++idx) {
                  const uint32_t b = active_buckets[idx];
                  ReverseEdge* span_begin =
                      reverse_edges.data() + bucket_offsets[b];
                  ReverseEdge* span_end = span_begin + bucket_counts[b];
                  std::sort(span_begin, span_end,
                            [](const ReverseEdge& x, const ReverseEdge& y) {
                              return std::tie(x.level, x.dest, x.src) <
                                     std::tie(y.level, y.dest, y.src);
                            });
                  const size_t span_len = bucket_counts[b];
                  size_t pos = 0;
                  while (pos < span_len) {
                    const uint32_t dest = span_begin[pos].dest;
                    const uint8_t lvl = span_begin[pos].level;
                    const size_t max_deg =
                        (lvl == 0 ? 2 : 1) * static_cast<size_t>(options.m);
                    size_t group_end = pos + 1;
                    while (group_end < span_len &&
                           span_begin[group_end].dest == dest &&
                           span_begin[group_end].level == lvl) {
                      ++group_end;
                    }
                    const size_t s = node_slot[dest] + lvl;
                    uint32_t* base_ptr = slot_edges.data() + slot_offset[s];
                    size_t len = slot_size[s];
                    if (len + (group_end - pos) <= max_deg) {
                      for (; pos < group_end; ++pos) {
                        const uint32_t src = span_begin[pos].src;
                        if (src != dest &&
                            std::find(base_ptr, base_ptr + len, src) ==
                                base_ptr + len) {
                          base_ptr[len++] = src;
                        }
                      }
                      slot_size[s] = static_cast<uint32_t>(len);
                      continue;
                    }
                    ws.merged.assign(base_ptr, base_ptr + len);
                    for (; pos < group_end; ++pos) {
                      const uint32_t src = span_begin[pos].src;
                      if (src != dest &&
                          std::find(ws.merged.begin(), ws.merged.end(), src) ==
                              ws.merged.end()) {
                        ws.merged.push_back(src);
                      }
                    }
                    if (ws.merged.size() <= max_deg) {
                      write_neighbors(dest, lvl, ws.merged);
                      continue;
                    }
                    score_nodes(dest, ws.merged.data(), ws.merged.size(),
                                ws.distances);
                    ws.scored.resize(ws.merged.size());
                    for (size_t i = 0; i < ws.merged.size(); ++i) {
                      ws.scored[i] = {ws.distances[i], ws.merged[i]};
                    }
                    std::sort(ws.scored.begin(), ws.scored.end());
                    const size_t pruned_deg =
                        std::max<size_t>(1, max_deg - max_deg / 5);
                    select_neighbors(ws.scored, pruned_deg, ws.selected,
                                     ws.rejected);
                    write_neighbors(dest, lvl, ws.selected);
                  }
                }
              })) {
        return ynn_status_error;
      }

      completed += step;
      if (progress) progress(completed, num_nodes);
    }
  }

  graph.base_offsets.reserve(num_nodes + 1);
  graph.base_offsets.push_back(0);
  for (size_t node = 0; node < num_nodes; ++node) {
    const uint32_t* edges = nullptr;
    size_t count = 0;
    slot_range(static_cast<uint32_t>(node), 0, &edges, &count);
    graph.base_edges.insert(graph.base_edges.end(), edges, edges + count);
    if (graph.base_edges.size() > std::numeric_limits<uint32_t>::max()) {
      return ynn_status_unsupported_parameter;
    }
    graph.base_offsets.push_back(static_cast<uint32_t>(graph.base_edges.size()));
  }
  graph.upper.resize(maximum_level);
  for (size_t level = 1; level <= maximum_level; ++level) {
    hnsw_layer& destination = graph.upper[level - 1];
    destination.offsets.push_back(0);
    for (size_t node = 0; node < num_nodes; ++node) {
      if (graph.levels[node] < level) continue;
      const uint32_t* edges = nullptr;
      size_t count = 0;
      slot_range(static_cast<uint32_t>(node), level, &edges, &count);
      destination.nodes.push_back(static_cast<uint32_t>(node));
      destination.edges.insert(destination.edges.end(), edges, edges + count);
      if (destination.edges.size() > std::numeric_limits<uint32_t>::max()) {
        return ynn_status_unsupported_parameter;
      }
      destination.offsets.push_back(
          static_cast<uint32_t>(destination.edges.size()));
    }
  }
  if (!validate_hnsw_graph(graph)) return ynn_status_error;
  *out_graph = std::move(graph);
  return ynn_status_success;
}

ynn_status build_hnsw_graph(const void* vectors, bool vectors_bf16,
                            size_t num_nodes, size_t width,
                            const hnsw_build_options& options,
                            hnsw_graph* out_graph, ynn_threadpool_t threadpool,
                            hnsw_progress_fn progress) {
  return build_hnsw_graph(vectors, vectors_bf16, num_nodes, width, options,
                          out_graph, std::move(progress),
                          make_threadpool_parallel_for(threadpool));
}

ynn_status prepare_hnsw_dense_search(const hnsw_graph& source,
                                     const void* vectors, bool vectors_bf16,
                                     hnsw_search_index* out_index) {
  if (vectors == nullptr || out_index == nullptr ||
      !validate_hnsw_graph(source)) {
    return ynn_status_invalid_parameter;
  }
  *out_index = {};
  build_reordered_graph(source, &out_index->graph);
  out_index->vectors_bf16 = vectors_bf16;
  out_index->kernels = get_hnsw_dot_kernels(vectors_bf16);
  const size_t nodes = out_index->graph.labels.size();
  const size_t row_bytes =
      static_cast<size_t>(source.width) *
      (vectors_bf16 ? sizeof(uint16_t) : sizeof(float));
  out_index->slice_bytes = row_bytes;
  out_index->packed_rows.resize(nodes * row_bytes);
  const auto* src_bytes = static_cast<const uint8_t*>(vectors);
  for (size_t new_u = 0; new_u < nodes; ++new_u) {
    const size_t label = out_index->graph.labels[new_u];
    std::memcpy(out_index->packed_rows.data() + new_u * row_bytes,
                src_bytes + label * row_bytes, row_bytes);
  }
  return ynn_status_success;
}

ynn_status prepare_hnsw_quantized_search(
    const hnsw_graph& source, const void* weights, const void* scales,
    bool scale_fp16, const void* zeros, bool zero_fp16, int32_t bits,
    int32_t group_size, float input_scale, hnsw_search_index* out_index) {
  const bool has_zero = zeros != nullptr;
  if (weights == nullptr || scales == nullptr || out_index == nullptr ||
      (bits != 2 && bits != 4 && bits != 8) || !std::isfinite(input_scale) ||
      input_scale < 0.0f || group_size <= 0 ||
      source.width % static_cast<uint32_t>(group_size) != 0 ||
      group_size % 32 != 0 || (has_zero && bits != 4) ||
      !validate_hnsw_graph(source)) {
    return ynn_status_invalid_parameter;
  }
  *out_index = {};
  build_reordered_graph(source, &out_index->graph);
  out_index->bits = bits;
  out_index->group_size = group_size;
  out_index->input_scale = input_scale;
  out_index->scale_fp16 = scale_fp16;
  out_index->zero_fp16 = zero_fp16;
  out_index->has_zero = has_zero;
  out_index->slice_bytes = hnsw_quantized_slice_bytes(
      source.width, bits, group_size, scale_fp16, zero_fp16, has_zero);
  const size_t nodes = out_index->graph.labels.size();
  out_index->packed_rows.resize(nodes * out_index->slice_bytes);

  const size_t row_bytes =
      static_cast<size_t>(source.width) / (8u / static_cast<size_t>(bits));
  const size_t B = static_cast<size_t>(source.width) /
                   static_cast<size_t>(group_size);
  const size_t s_row_bytes = B * (scale_fp16 ? 2u : 4u);
  const size_t z_row_bytes = has_zero ? B * (zero_fp16 ? 2u : 4u) : 0u;
  const auto* w_bytes = static_cast<const uint8_t*>(weights);
  const auto* s_bytes = static_cast<const uint8_t*>(scales);
  const auto* z_bytes = static_cast<const uint8_t*>(zeros);
  for (size_t new_u = 0; new_u < nodes; ++new_u) {
    const size_t label = out_index->graph.labels[new_u];
    hnsw_pack_quantized_slice(
        out_index->packed_rows.data() + new_u * out_index->slice_bytes,
        w_bytes + label * row_bytes, s_bytes + label * s_row_bytes,
        has_zero ? z_bytes + label * z_row_bytes : nullptr, source.width, bits,
        group_size, scale_fp16, zero_fp16, has_zero);
  }
  return ynn_status_success;
}

ynn_status search_hnsw(const hnsw_search_index& index, const float* query,
                       size_t query_width, size_t k, size_t ef,
                       hnsw_workspace* workspace,
                       const hnsw_neighbor** out_neighbors, size_t* out_count,
                       hnsw_parallel_for_fn parallel_for) {
  const hnsw_graph& graph = index.graph;
  const size_t nodes = graph.labels.size();
  if (query == nullptr || workspace == nullptr || out_neighbors == nullptr ||
      out_count == nullptr || nodes == 0 || query_width != graph.width ||
      k == 0 || k > nodes || ef < k) {
    return ynn_status_invalid_parameter;
  }
  for (size_t i = 0; i < query_width; ++i) {
    if (!std::isfinite(query[i])) return ynn_status_invalid_parameter;
  }

  if (index.bits != 0) {
    hnsw_prepare_quantized_query(query, graph.width, index.bits,
                                 index.group_size, index.input_scale,
                                 workspace->q_i8, workspace->q_scale,
                                 workspace->q_bsum);
  }
  ef = std::min(ef, nodes);
  const size_t visited_words = (nodes + 63u) / 64u;
  workspace->visited.assign(visited_words, 0);
  ++workspace->epoch;
  workspace->frontier.reserve(ef * 2 + 1);
  workspace->nearest.reserve(ef + 1);
  workspace->unseen.reserve(256);
  workspace->unseen_distances.reserve(256);
  workspace->frontier.clear();
  workspace->nearest.clear();
  workspace->result.clear();
  workspace->result.reserve(k);

  std::atomic<bool> nonfinite{false};
  const auto slice_ptr = [&](uint32_t node) noexcept -> const uint8_t* {
    return index.packed_rows.data() +
           static_cast<size_t>(node) * index.slice_bytes;
  };
  const auto single_distance = [&](uint32_t node) noexcept -> float {
    const uint8_t* slice = slice_ptr(node);
    const float score =
        index.bits == 0
            ? index.kernels.dot(query, slice, graph.width)
            : hnsw_quantized_slice_dot(
                  slice, graph.width, index.bits, index.group_size,
                  index.scale_fp16, index.zero_fp16, index.has_zero,
                  workspace->q_i8.data(), workspace->q_scale.data(),
                  workspace->q_bsum.data());
    const float d = 1.0f - score;
    if (!std::isfinite(d)) {
      nonfinite.store(true, std::memory_order_relaxed);
      return 0.0f;
    }
    return d;
  };
  const auto score_range = [&](const uint32_t* batch, float* distances,
                               size_t begin, size_t end) noexcept {
    size_t idx = begin;
    for (; idx + 4 <= end; idx += 4) {
      if (idx + 8 <= end) {
        hnsw_prefetch_slice(slice_ptr(batch[idx + 4]), index.slice_bytes);
        hnsw_prefetch_slice(slice_ptr(batch[idx + 5]), index.slice_bytes);
        hnsw_prefetch_slice(slice_ptr(batch[idx + 6]), index.slice_bytes);
        hnsw_prefetch_slice(slice_ptr(batch[idx + 7]), index.slice_bytes);
      }
      const uint8_t* s0 = slice_ptr(batch[idx + 0]);
      const uint8_t* s1 = slice_ptr(batch[idx + 1]);
      const uint8_t* s2 = slice_ptr(batch[idx + 2]);
      const uint8_t* s3 = slice_ptr(batch[idx + 3]);
      float dots[4];
      if (index.bits == 0) {
        index.kernels.dot4(query, s0, s1, s2, s3, graph.width, dots);
      } else {
        hnsw_quantized_slices4_dot(
            s0, s1, s2, s3, graph.width, index.bits, index.group_size,
            index.scale_fp16, index.zero_fp16, index.has_zero,
            workspace->q_i8.data(), workspace->q_scale.data(),
            workspace->q_bsum.data(), dots);
      }
      for (size_t r = 0; r < 4; ++r) {
        const float d = 1.0f - dots[r];
        if (!std::isfinite(d)) {
          nonfinite.store(true, std::memory_order_relaxed);
          distances[idx + r] = 0.0f;
        } else {
          distances[idx + r] = d;
        }
      }
    }
    for (; idx < end; ++idx) distances[idx] = single_distance(batch[idx]);
  };
  const auto score_batch = [&](const uint32_t* batch, size_t count,
                               float* distances) -> bool {
    if (parallel_for && count >= 48) {
      const size_t chunk = std::max<size_t>(16, ((count / 4) + 3u) & ~size_t{3});
      const size_t tasks = (count + chunk - 1) / chunk;
      if (tasks >= 2) {
        if (!parallel_for(tasks, [&](size_t task) {
              const size_t begin = task * chunk;
              if (begin >= count || nonfinite.load(std::memory_order_relaxed)) {
                return;
              }
              score_range(batch, distances, begin, std::min(count, begin + chunk));
            })) {
          return false;
        }
        return !nonfinite.load(std::memory_order_relaxed);
      }
    }
    score_range(batch, distances, 0, count);
    return !nonfinite.load(std::memory_order_relaxed);
  };

  uint32_t entry = graph.entry;
  float best_entry = single_distance(entry);
  if (nonfinite.load(std::memory_order_relaxed)) return ynn_status_error;
  for (size_t level = graph.upper.size(); level != 0; --level) {
    bool changed = true;
    while (changed) {
      changed = false;
      const uint32_t* neighbors = nullptr;
      size_t count = 0;
      graph_neighbors(graph, entry, level, &neighbors, &count);
      workspace->unseen_distances.resize(count);
      score_range(neighbors, workspace->unseen_distances.data(), 0, count);
      if (nonfinite.load(std::memory_order_relaxed)) return ynn_status_error;
      for (size_t idx = 0; idx < count; ++idx) {
        const float value = workspace->unseen_distances[idx];
        if (value < best_entry) {
          best_entry = value;
          entry = neighbors[idx];
          changed = true;
        }
      }
    }
  }

  heap_push(workspace->frontier, -best_entry, entry);
  workspace->nearest.emplace_back(best_entry, entry);
  workspace->visited[entry >> 6u] |= (uint64_t{1} << (entry & 63u));
  bool nearest_heap = ef == 1;
  float best =
      nearest_heap ? best_entry : std::numeric_limits<float>::infinity();
  while (!workspace->frontier.empty()) {
    if (nearest_heap && -workspace->frontier.front().first > best) break;
    workspace->unseen.clear();
    const size_t max_pops = parallel_for ? (nearest_heap ? 2 : 4) : 1;
    for (size_t pop_idx = 0;
         pop_idx < max_pops && !workspace->frontier.empty(); ++pop_idx) {
      if (nearest_heap && -workspace->frontier.front().first > best) break;
      const uint32_t current = heap_pop(workspace->frontier).second;
      const uint32_t* neighbors = nullptr;
      size_t count = 0;
      graph_neighbors(graph, current, 0, &neighbors, &count);
      for (size_t i = 0; i < count; ++i) {
        const uint32_t nb = neighbors[i];
        const uint64_t mask = uint64_t{1} << (nb & 63u);
        uint64_t& word = workspace->visited[nb >> 6u];
        if ((word & mask) == 0) {
          word |= mask;
          workspace->unseen.push_back(nb);
        }
      }
      if (nearest_heap && workspace->unseen.size() >= 48) break;
      if (!nearest_heap &&
          workspace->nearest.size() + workspace->unseen.size() >= ef) {
        break;
      }
    }
    if (workspace->unseen.empty()) continue;
    workspace->unseen_distances.resize(workspace->unseen.size());
    if (!score_batch(workspace->unseen.data(), workspace->unseen.size(),
                     workspace->unseen_distances.data())) {
      return ynn_status_error;
    }
    for (size_t idx = 0; idx < workspace->unseen.size(); ++idx) {
      const uint32_t nb = workspace->unseen[idx];
      const float value = workspace->unseen_distances[idx];
      if (!nearest_heap) {
        workspace->nearest.emplace_back(value, nb);
        if (workspace->nearest.size() == ef) {
          std::make_heap(workspace->nearest.begin(), workspace->nearest.end(),
                         ByDistance{});
          nearest_heap = true;
          best = workspace->nearest.front().first;
        }
        workspace->unseen_distances[idx] = value;
      } else if (value < best) {
        std::pop_heap(workspace->nearest.begin(), workspace->nearest.end(),
                      ByDistance{});
        workspace->nearest.back() = {value, nb};
        std::push_heap(workspace->nearest.begin(), workspace->nearest.end(),
                       ByDistance{});
        best = workspace->nearest.front().first;
      } else {
        workspace->unseen_distances[idx] =
            std::numeric_limits<float>::infinity();
      }
    }
    for (size_t idx = 0; idx < workspace->unseen.size(); ++idx) {
      const float value = workspace->unseen_distances[idx];
      if (!nearest_heap || value <= best) {
        heap_push(workspace->frontier, -value, workspace->unseen[idx]);
      }
    }
  }
  if (workspace->nearest.size() < k) return ynn_status_error;
  std::partial_sort(
      workspace->nearest.begin(),
      workspace->nearest.begin() + static_cast<ptrdiff_t>(k),
      workspace->nearest.end(),
      [&](const std::pair<float, uint32_t>& left,
          const std::pair<float, uint32_t>& right) {
        return left.first < right.first ||
               (left.first == right.first &&
                graph.labels[left.second] < graph.labels[right.second]);
      });
  for (size_t rank = 0; rank < k; ++rank) {
    const auto& item = workspace->nearest[rank];
    workspace->result.push_back({item.first, graph.labels[item.second]});
  }
  *out_neighbors = workspace->result.data();
  *out_count = workspace->result.size();
  return ynn_status_success;
}

ynn_status search_hnsw(const hnsw_search_index& index, const float* query,
                       size_t query_width, size_t k, size_t ef,
                       hnsw_workspace* workspace,
                       const hnsw_neighbor** out_neighbors, size_t* out_count,
                       ynn_threadpool_t threadpool) {
  return search_hnsw(index, query, query_width, k, ef, workspace, out_neighbors,
                     out_count, make_threadpool_parallel_for(threadpool));
}

}  // namespace ynn
