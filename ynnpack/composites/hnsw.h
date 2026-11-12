// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_COMPOSITES_HNSW_H_
#define XNNPACK_YNNPACK_COMPOSITES_HNSW_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "ynnpack/include/ynnpack.h"

namespace ynn {

constexpr uint32_t kHnswMaxNodes = 1u << 24;

struct hnsw_layer {
  std::vector<uint32_t> nodes;
  std::vector<uint32_t> offsets;
  std::vector<uint32_t> edges;
};

struct hnsw_graph {
  uint32_t width = 0;
  uint32_t m = 0;
  uint32_t entry = 0;
  std::vector<uint32_t> labels;
  std::vector<uint32_t> base_offsets;
  std::vector<uint32_t> base_edges;
  std::vector<uint8_t> levels;
  std::vector<hnsw_layer> upper;

  size_t num_nodes() const noexcept { return labels.size(); }
  size_t runtime_bytes() const noexcept;
};

struct hnsw_build_options {
  uint32_t m = 32;
  size_t ef_construction = 200;
  uint64_t seed = 42;
};

struct hnsw_neighbor {
  float distance;
  uint32_t id;
};

struct hnsw_workspace {
  std::vector<uint64_t> visited;
  uint32_t epoch = 0;
  std::vector<uint32_t> unseen;
  std::vector<float> unseen_distances;
  std::vector<std::pair<float, uint32_t>> frontier;
  std::vector<std::pair<float, uint32_t>> nearest;
  std::vector<hnsw_neighbor> result;
  std::vector<int8_t> q_i8;
  std::vector<float> q_scale;
  std::vector<float> q_bsum;
};

struct hnsw_dot_kernels {
  using dot_fn = float (*)(const float* query, const void* row, size_t width);
  using dot4_fn = void (*)(const float* query, const void* r0, const void* r1,
                           const void* r2, const void* r3, size_t width,
                           float out[4]);
  using row_dot_fn = float (*)(const void* left, const void* right,
                               size_t width);
  using row_dot4_fn = void (*)(const void* query, const void* r0,
                               const void* r1, const void* r2, const void* r3,
                               size_t width, float out[4]);

  dot_fn dot = nullptr;
  dot4_fn dot4 = nullptr;
  row_dot_fn row_dot = nullptr;
  row_dot4_fn row_dot4 = nullptr;
};

struct hnsw_search_index {
  hnsw_graph graph;
  std::vector<uint8_t> packed_rows;
  int32_t bits = 0;
  int32_t group_size = 0;
  float input_scale = 0.0f;
  bool vectors_bf16 = false;
  bool scale_fp16 = false;
  bool zero_fp16 = false;
  bool has_zero = false;
  size_t slice_bytes = 0;
  hnsw_dot_kernels kernels;
};

using hnsw_progress_fn = std::function<void(size_t completed, size_t total)>;
using hnsw_parallel_for_fn =
    std::function<bool(size_t task_count, std::function<void(size_t)> body)>;

// Low-level HNSW distance & quantized-slice microkernels.
hnsw_dot_kernels get_hnsw_dot_kernels(bool bf16) noexcept;
float hnsw_weight_element_f32(const void* row, size_t column,
                              bool bf16) noexcept;
void hnsw_prefetch_slice(const void* slice_ptr, size_t slice_bytes) noexcept;
size_t hnsw_quantized_slice_bytes(size_t width, int32_t bits,
                                  int32_t group_size, bool scale_fp16,
                                  bool zero_fp16, bool has_zero) noexcept;
void hnsw_pack_quantized_slice(uint8_t* dst, const void* w_row,
                               const void* s_row, const void* z_row,
                               size_t width, int32_t bits, int32_t group_size,
                               bool scale_fp16, bool zero_fp16,
                               bool has_zero) noexcept;
void hnsw_prepare_quantized_query(const float* query, size_t width,
                                  int32_t bits, int32_t group_size,
                                  float input_scale, std::vector<int8_t>& q_i8,
                                  std::vector<float>& q_scale,
                                  std::vector<float>& q_bsum);
float hnsw_quantized_slice_dot(const uint8_t* slice_ptr, size_t width,
                               int32_t bits, int32_t group_size,
                               bool scale_fp16, bool zero_fp16, bool has_zero,
                               const int8_t* q_i8, const float* q_scale,
                               const float* q_bsum) noexcept;
void hnsw_quantized_slices4_dot(const uint8_t* s0, const uint8_t* s1,
                                const uint8_t* s2, const uint8_t* s3,
                                size_t width, int32_t bits, int32_t group_size,
                                bool scale_fp16, bool zero_fp16, bool has_zero,
                                const int8_t* q_i8, const float* q_scale,
                                const float* q_bsum, float out[4]) noexcept;

// HNSW topology validation, deterministic construction, layout packing & search.
bool validate_hnsw_graph(const hnsw_graph& graph);

ynn_status build_hnsw_graph(const void* vectors, bool vectors_bf16,
                            size_t num_nodes, size_t width,
                            const hnsw_build_options& options,
                            hnsw_graph* out_graph,
                            hnsw_progress_fn progress = {},
                            hnsw_parallel_for_fn parallel_for = {});

ynn_status build_hnsw_graph(const void* vectors, bool vectors_bf16,
                            size_t num_nodes, size_t width,
                            const hnsw_build_options& options,
                            hnsw_graph* out_graph, ynn_threadpool_t threadpool,
                            hnsw_progress_fn progress = {});

ynn_status prepare_hnsw_dense_search(const hnsw_graph& source,
                                     const void* vectors, bool vectors_bf16,
                                     hnsw_search_index* out_index);

ynn_status prepare_hnsw_quantized_search(
    const hnsw_graph& source, const void* weights, const void* scales,
    bool scale_fp16, const void* zeros, bool zero_fp16, int32_t bits,
    int32_t group_size, float input_scale, hnsw_search_index* out_index);

ynn_status search_hnsw(const hnsw_search_index& index, const float* query,
                       size_t query_width, size_t k, size_t ef,
                       hnsw_workspace* workspace,
                       const hnsw_neighbor** out_neighbors, size_t* out_count,
                       hnsw_parallel_for_fn parallel_for = {});

ynn_status search_hnsw(const hnsw_search_index& index, const float* query,
                       size_t query_width, size_t k, size_t ef,
                       hnsw_workspace* workspace,
                       const hnsw_neighbor** out_neighbors, size_t* out_count,
                       ynn_threadpool_t threadpool);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_COMPOSITES_HNSW_H_
