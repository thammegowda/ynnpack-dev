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
void quantize_blockwise_blocks_f32(int32_t bits, size_t m, size_t k, size_t block_size, bool compute_bsum,
                                   const float* input, int8_t* quantized_input, float* input_scales, float* input_bsums,
                                   size_t block_start, size_t block_end);
void blockwise_linear_f32(int32_t bits, size_t m, size_t n, size_t k, size_t block_size, bool has_zero,
                          float output_scale, const void* packed_tiles, const int8_t* quantized_input,
                          const float* input_scales, const float* input_bsums, float* output, size_t tile_start,
                          size_t tile_end);

bool is_symmetric_linear_supported(int32_t bits);
size_t symmetric_linear_padded_block(size_t block_size);
size_t packed_symmetric_linear_size(int32_t bits, size_t n, size_t k, size_t block_size);
void pack_symmetric_linear_tiles(int32_t bits, size_t n, size_t k, size_t row_offset, size_t block_size,
                                 const void* weights, const void* scales, bool scales_fp16, size_t tile_start,
                                 size_t tile_end, void* packed_tiles);
void quantize_symmetric_blocks_f32(int32_t bits, size_t m, size_t k, size_t block_size, float static_scale,
                                   const float* input, int8_t* quantized_input, float* input_scales, size_t block_start,
                                   size_t block_end);
void symmetric_linear_f32(int32_t bits, size_t m, size_t n, size_t k, size_t block_size, float output_scale,
                          const void* packed_tiles, const int8_t* quantized_input, const float* input_scales,
                          float* output, size_t output_stride, size_t tile_start, size_t tile_end);

struct dense_linear_plan {
    ynn_type a_type = ynn_type_invalid;
    ynn_type b_type = ynn_type_invalid;
    ynn_type c_type = ynn_type_invalid;
    size_t n = 0;
    size_t k = 0;
    size_t block_n = 0;
    size_t tile_k = 0;
    size_t tiles_k = 0;
    size_t blocks_n = 0;
    size_t tile_bytes = 0;
    size_t total_bytes = 0;
};

ynn_status make_dense_linear_plan(ynn_type a_type, ynn_type b_type, ynn_type c_type, size_t n, size_t k,
                                  dense_linear_plan* out_plan);
void pack_dense_linear_tiles(const dense_linear_plan& plan, bool transpose_b, const void* b_ptr,
                             size_t b_row_stride_bytes, size_t tile_start, size_t tile_end, void* packed_tiles,
                             int32_t* col_sums = nullptr);
bool dense_linear_select_kernel(const dense_linear_plan& plan, size_t m, bool* out_transpose_a, size_t* out_tile_m,
                                size_t* out_a_packed_bytes);
void dense_linear_pack_a(const dense_linear_plan& plan, size_t m, size_t tile_m, const void* a_ptr,
                         size_t a_row_stride_bytes, void* a_packed);
void dense_linear_run_tiles(const dense_linear_plan& plan, size_t m, bool transposed_a, size_t tile_m,
                            const void* a_ptr, size_t a_row_stride_bytes, const void* packed_b, const float* bias_f32,
                            float* output_f32, size_t output_stride_elems, size_t tile_start, size_t tile_end);
void quantize_dynamic_int8_rows_f32(size_t rows, size_t k, const float* input, int8_t* quantized, float* scales,
                                    int32_t* zero_points, size_t row_start, size_t row_end);
size_t quantized_linear_scratch_bytes(const dense_linear_plan& plan, size_t m, size_t tile_m);
void quantized_linear_run_tiles(const dense_linear_plan& plan, size_t m, bool transposed_a, size_t tile_m,
                                const void* a_ptr, size_t a_row_stride_bytes, const void* packed_b,
                                const int32_t* b_col_sums, const float* a_scales, const int32_t* a_zero_points,
                                const float* b_scales, const float* bias_f32, float* output_f32,
                                size_t output_stride_elems, int32_t* i32_scratch, size_t tile_start, size_t tile_end);
void convert_f32_to_bf16_rows(size_t rows, size_t cols, const float* src, size_t src_stride_elems, uint16_t* dst,
                              size_t dst_stride_elems, size_t row_start, size_t row_end);
void convert_bf16_to_f32_rows(size_t rows, size_t cols, const uint16_t* src, size_t src_stride_elems, float* dst,
                              size_t dst_stride_elems, size_t row_start, size_t row_end);

void layer_norm_f32(size_t rows, size_t width, float epsilon, const float* input, const float* residual_in,
                    const float* weight, const float* bias, float* output_norm, float* output_residual,
                    size_t row_start, size_t row_end);
void softmax_f32(size_t rows, size_t width, bool logarithmic, const float* input, float* output, size_t row_start,
                 size_t row_end);
void gelu_f32(size_t count, bool approximate, const float* input, float* output, size_t start, size_t end);
void tanh_f32(size_t count, const float* input, float* output, size_t start, size_t end);
void static_round_f32(size_t count, float scale, const float* input, float* output, size_t start, size_t end);
void cast_elements(ynn_type src_type, ynn_type dst_type, size_t count, const void* src, void* dst, size_t start,
                   size_t end);
void broadcast_binary_f32(bool is_multiply, size_t rank, const size_t* out_shape, const size_t* a_strides,
                          const size_t* b_strides, size_t out_elements, const float* a, const float* b, float* out,
                          size_t start, size_t end);
void transpose_elements(size_t elem_bytes, size_t rank, const size_t* in_shape, const size_t* out_strides_elems,
                        const uint8_t* src, uint8_t* dst, size_t outer_start, size_t outer_end);

void transpose_packed_subbyte_tile(int32_t bits, const uint8_t* source, size_t source_stride_bytes, size_t row_start,
                                   size_t row_count, size_t col_start, size_t col_count, uint8_t* destination);
size_t gated_activation_tile_count(size_t rows, size_t width);
ynn_status silu_gate_f32(size_t rows, size_t width, const float* input, float* output, size_t row_start,
                         size_t row_end);
ynn_status silu_gate_tiles_f32(size_t rows, size_t width, const float* input, float* output, size_t tile_start,
                               size_t tile_end);
ynn_status gelu_gate_f32(size_t rows, size_t width, const float* input, float* output, size_t row_start,
                         size_t row_end);
ynn_status gelu_gate_tiles_f32(size_t rows, size_t width, const float* input, float* output, size_t tile_start,
                               size_t tile_end);
size_t gated_multiply_tile_count(size_t elements);
ynn_status silu_multiply_tiles_f32(size_t elements, const float* gate, const float* up, float* output,
                                   size_t tile_start, size_t tile_end);
ynn_status gelu_multiply_tiles_f32(size_t elements, const float* gate, const float* up, float* output,
                                   size_t tile_start, size_t tile_end);
ynn_status packed_ffn_mid_tiles_f32(int32_t bits, size_t m, size_t input_size, size_t intermediate_size,
                                    float gate_output_scale, float down_input_scale, bool silu, const void* gate_packed,
                                    const void* up_packed, const int8_t* quantized_input, const float* input_scales,
                                    int8_t* quantized_mid, size_t tile_start, size_t tile_end);
void elementwise_binary_f32(bool is_multiply, size_t elements, size_t rhs_stride, const float* lhs, const float* rhs,
                            float* output, size_t start, size_t end);

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
void argmax_chunk_f32(size_t count, const float* row_logits, size_t col_offset, float* out_max, int32_t* out_index,
                      bool* out_invalid);
void argmax_rows_f32(size_t rows, size_t width, const float* logits, int32_t* tokens);

} // namespace ynn

#endif // XNNPACK_YNNPACK_COMPOSITES_COMPOSITES_H_
