// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_COMPOSITES_COMPOSITES_H_
#define XNNPACK_YNNPACK_COMPOSITES_COMPOSITES_H_

#include <cstddef>
#include <cstdint>

#include "ynnpack/composites/hnsw.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

// This header defines helpers for implementing common higher level operations
// using YNNPACK's lower level public API.

struct blockwise_quantization_f32_params {
    const float* scales;
    const int32_t* zero_points;
    size_t block_size;
    size_t num_blocks;
};

struct quantized_attention_f32_params {
    size_t batch_size;
    size_t query_length;
    size_t query_heads;
    size_t key_value_heads;
    size_t key_start;
    size_t key_length;
    size_t head_dim;
    size_t key_batch_stride;
    size_t mask_batch_stride;
    size_t mask_head_stride;
    size_t mask_query_stride;
    size_t mask_key_stride;
    float scale;
    blockwise_quantization_f32_params key_quantization;
    blockwise_quantization_f32_params value_quantization;
};

size_t quantized_attention_f32_scratch_size(const quantized_attention_f32_params& params);
size_t quantized_attention_f32_work_items(const quantized_attention_f32_params& params);
void quantized_attention_f32(const quantized_attention_f32_params& params, const float* query, const int8_t* key,
                             const int8_t* value, const float* mask, float* output, size_t row_start, size_t row_end,
                             void* scratch);

struct attention_f32_params {
    size_t batch_size;
    size_t query_length;
    size_t query_heads;
    size_t key_value_heads;
    size_t key_start;
    size_t key_length;
    size_t total_keys;
    size_t head_dim;
    size_t key_batch_stride;
    size_t mask_batch_stride;
    size_t mask_head_stride;
    size_t mask_query_stride;
    size_t mask_key_stride;
    float scale;
};

size_t attention_f32_work_items(const attention_f32_params& params);
size_t attention_f32_scratch_size(const attention_f32_params& params);
void attention_f32(const attention_f32_params& params, const float* query, const float* key, const float* value,
                   const float* mask, float* output, size_t item_start, size_t item_end, float* scratch);

bool is_blockwise_linear_supported(int32_t bits);
size_t blockwise_linear_tile_count(size_t n);
size_t packed_blockwise_linear_size(int32_t bits, size_t n, size_t k, size_t block_size, bool has_zero);
void pack_blockwise_linear_tiles(int32_t bits, size_t n, size_t k, size_t block_size, const void* weights,
                                 const void* scales, bool scales_fp16, const void* zeros, bool zeros_fp16,
                                 size_t tile_start, size_t tile_end, void* packed_tiles);
void quantize_blockwise_activations_f32(int32_t bits, size_t m, size_t k, size_t block_size, bool compute_bsum,
                                        const float* input, int8_t* quantized_input, float* input_scales,
                                        float* input_bsums, size_t row_start, size_t row_end);
void blockwise_linear_f32(int32_t bits, size_t m, size_t n, size_t k, size_t block_size, bool has_zero,
                          float output_scale, const void* packed_tiles, const int8_t* quantized_input,
                          const float* input_scales, const float* input_bsums, float* output, size_t tile_start,
                          size_t tile_end);

void transpose_packed_subbyte_tile(int32_t bits, const uint8_t* source, size_t source_stride_bytes, size_t row_start,
                                   size_t row_count, size_t col_start, size_t col_count, uint8_t* destination);
ynn_status silu_gate_f32(size_t rows, size_t width, const float* input, float* output, size_t row_start,
                         size_t row_end);
ynn_status gelu_gate_f32(size_t rows, size_t width, const float* input, float* output, size_t row_start,
                         size_t row_end);

void rms_norm_f32(size_t rows, size_t width, float epsilon, const float* input, const float* weight,
                  const float* residual, bool has_scale, float scale, float* output, size_t row_start, size_t row_end);
void rotary_f32(size_t sequence_rows, size_t heads, size_t width, bool shared_angles, const float* input,
                const float* cosine, const float* sine, float* output, size_t row_start, size_t row_end);
void rms_rotary_f32(size_t sequence_rows, size_t heads, size_t width, bool sliced, size_t input_rows,
                    size_t input_stride, size_t sequence_offset, size_t channel_offset, float epsilon,
                    bool shared_angles, const float* input, const float* weight, const float* cosine, const float* sine,
                    float* output, size_t vec_start, size_t vec_end);

void dequantize_embedding_f32(const int32_t* tokens, size_t num_tokens, const void* weights, bool weights_bf16,
                              int32_t packed_bits, bool unsigned_packed, size_t width, size_t groups,
                              const void* scales, bool scales_fp16, const void* zeros, bool zeros_fp16,
                              float multiplier, float* output, size_t row_start, size_t row_end);
void quantize_symmetric_int8_f32(size_t count, float scale, const float* input, int8_t* output);
void argmax_rows_f32(size_t rows, size_t width, const float* logits, int32_t* tokens);

// gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
ynn_status define_gelu(ynn_subgraph_t subgraph, uint32_t input_id, uint32_t& output_id);

// approx_gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
ynn_status define_approx_gelu(ynn_subgraph_t subgraph, uint32_t input_id, uint32_t& output_id);

// elu(x) = x >= 0 ? x : alpha * (exp(x) - 1)
ynn_status define_elu(ynn_subgraph_t subgraph, uint32_t input_id, float alpha, uint32_t& output_id);

// leaky_relu(x) = x >= 0 ? x : alpha * x
ynn_status define_leaky_relu(ynn_subgraph_t subgraph, uint32_t input_id, float alpha, uint32_t& output_id);

// hardswish(x) = x * clamp(x / 6 + 0.5, 0, 1)
ynn_status define_hardswish(ynn_subgraph_t subgraph, uint32_t input_id, uint32_t& output_id);

// softmax(x)_i = exp(beta * x_i) / sum_j(exp(beta * x_j))
ynn_status define_softmax(ynn_subgraph_t subgraph, uint32_t input_id, float beta, uint32_t& output_id);

// log_softmax(x)_i = x_i - log(sum_j(exp(x_j)))
ynn_status define_log_softmax(ynn_subgraph_t subgraph, uint32_t input_id, uint32_t& output_id);

// Computes average pooling of a 2D buffer. The `input_id` and `output_id`
// values must refer to rank 4 tensors. When averaging, the number of samples is
// the number of samples that are not padding.
ynn_status define_average_pool_2d(ynn_subgraph_t subgraph, uint32_t input_id, ynn_type type, bool padding_same,
                                  size_t filter_height, size_t filter_width, size_t stride_height, size_t stride_width,
                                  uint32_t& output_id);

// Computes a sum reduction, optionally dividing by the number of elements in
// the reduction if `mean` is true. The quantization parameters may be
// `YNN_INVALID_VALUE_ID`, indicating an identity value of 0 (for zero point) or
// 1 (for scale).
ynn_status define_reduce_sum(ynn_subgraph_t subgraph, size_t num_axes, const int32_t* axes, uint32_t input_id,
                             uint32_t input_zero_point_id, uint32_t input_scale_id, bool keep_dims, bool mean,
                             bool squared, ynn_type output_type, uint32_t output_zero_point_id,
                             uint32_t output_scale_id, uint32_t& output_id);

// This function computes the quantization parameters of the result of a
// quantized dot operation. It computes the `zero_point` and `scale` values of
// the following equivalence:
//
//   (a.b - zero_point)*scale =
//       ((a - a_zero_point)*a_scale).(b - b_zerp_point)*b_scale
//
// It supports dynamic and static quantization parameters.
ynn_status define_dot_quantization(ynn_subgraph_t subgraph, size_t num_k_dims, uint32_t a_id, uint32_t a_zero_point_id,
                                   uint32_t a_scale_id, uint32_t b_id, uint32_t b_zero_point_id, uint32_t b_scale_id,
                                   uint32_t& zero_point_id, uint32_t& scale_id);

// Splits an f32 value into multiple bf16 values such that
// `sum(f32(output_ids[i])) ~= input_id`. The output values are ordered by
// descending significance.
ynn_status define_convert_f32_to_bf16_sum(ynn_subgraph_t subgraph, uint32_t input_id, size_t num_values,
                                          uint32_t* output_ids, uint32_t flags);

// Compute the dot of values that have been split into descending significance
// components. This computes the sum of the dot of the values in the cartesian
// product of a_ids x b_ids, where the sum of the index of the two values is
// not greater than `max_sum_index`.
ynn_status define_dot_sum(ynn_subgraph_t subgraph, size_t num_k_dims, size_t num_a_values, const uint32_t* a_ids,
                          size_t num_b_values, const uint32_t* b_ids, uint32_t input_c_id, uint32_t& output_id,
                          uint32_t flags, int max_sum_index = -1);

// Computes a blockwise quantized dot product of `a_id` and `b_id`.
// `a_id` has shape [..., M, K] (or [K] if rank 1).
// `b_id` has shape [..., K, N].
// `b_scale_id` is the scale tensor for `b_id` (either [..., N, num_blocks] or
// [..., num_blocks, 1, N]). `b_zero_point_id` is the zero point tensor for
// `b_id` (or YNN_INVALID_VALUE_ID). `a_zero_point_id` and `a_scale_id` are
// quantization parameters for `a_id` (or YNN_INVALID_VALUE_ID). `bias_id` is
// an optional bias tensor of shape [N] (or YNN_INVALID_VALUE_ID). `output_id`
// receives the result tensor.
ynn_status define_blockwise_dot(ynn_subgraph_t subgraph, uint32_t a_id, uint32_t a_zero_point_id, uint32_t a_scale_id,
                                uint32_t b_id, uint32_t b_zero_point_id, uint32_t b_scale_id, size_t block_size,
                                uint32_t bias_id, ynn_type output_type, uint32_t& output_id, uint32_t flags = 0);

// Computes a calibrated packed feed-forward layer in bounded intermediate
// tiles. If `gated` is true, the first weight contains gate rows followed by
// up rows and the activation is approx_gelu(gate) * up.
ynn_status define_packed_feed_forward(ynn_subgraph_t subgraph, uint32_t input_id, uint32_t input_scale_id,
                                      const uint32_t* gate_weight_ids, const uint32_t* gate_scale_ids,
                                      const uint32_t* up_weight_ids, const uint32_t* up_scale_ids,
                                      uint32_t first_output_scale_id, const uint32_t* down_weight_ids,
                                      uint32_t down_scale_id, uint32_t down_input_scale_id,
                                      uint32_t down_output_scale_id, size_t input_size, size_t intermediate_size,
                                      size_t tile_size, bool gated, uint32_t& output_id);

} // namespace ynn

#endif // XNNPACK_YNNPACK_COMPOSITES_COMPOSITES_H_
