// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_COMPOSITES_COMPOSITES_INTERNAL_H_
#define XNNPACK_YNNPACK_COMPOSITES_COMPOSITES_INTERNAL_H_

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ynnpack/base/half.h"
#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/hnsw.h"

#if defined(__clang__)
#define YNN_DISABLE_FP_CONTRACT _Pragma("clang fp contract(off)")
#else
#define YNN_DISABLE_FP_CONTRACT
#endif

#if defined(YNN_COMPOSITES_X86)
#define YNN_X86_TARGET_AVX2_FMA __attribute__((target("avx2,fma")))
#define YNN_X86_TARGET_AVX512 __attribute__((target("avx512f,avx512bw,avx512vl,fma")))
#define YNN_X86_TARGET_VNNI __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,fma,f16c")))
#define YNN_X86_TARGET_AMX \
    __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,fma,f16c,amx-tile,amx-int8")))
#if defined(__GNUC__) && !defined(__clang__)
#define YNN_X86_AMX_DPBUSD(dst, src1, src2)                                                                \
    __asm__ volatile(                                                                                      \
        "{tdpbusd\t%%tmm%c[_src2], %%tmm%c[_src1], %%tmm%c[_dst]|tdpbusd\t%%tmm%c[_dst], %%tmm%c[_src1], " \
        "%%tmm%c[_src2]}" ::[_dst] "i"(dst),                                                               \
        [_src1] "i"(src1), [_src2] "i"(src2)                                                               \
        : "memory")
#else
#define YNN_X86_AMX_DPBUSD(dst, src1, src2) _tile_dpbusd(dst, src1, src2)
#endif
#endif

#if defined(YNN_COMPOSITES_NEON)
#if defined(__clang__)
#define YNN_NEON_TARGET_BF16 __attribute__((target("bf16")))
#define YNN_NEON_TARGET_DOTPROD __attribute__((target("dotprod")))
#define YNN_NEON_TARGET_DOTPROD_FP16 __attribute__((target("dotprod,fp16")))
#elif defined(__GNUC__)
#define YNN_NEON_TARGET_BF16 __attribute__((target("+bf16")))
#define YNN_NEON_TARGET_DOTPROD __attribute__((target("+dotprod")))
#define YNN_NEON_TARGET_DOTPROD_FP16 __attribute__((target("+dotprod+fp16")))
#else
#define YNN_NEON_TARGET_BF16
#define YNN_NEON_TARGET_DOTPROD
#define YNN_NEON_TARGET_DOTPROD_FP16
#endif
#endif

namespace ynn::detail {

inline void prefetch_l1(const void* ptr) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(ptr, 0, 1);
#else
    (void)ptr;
#endif
}

// --- blockwise_linear variants (x86, neon) ---
#if defined(YNN_COMPOSITES_X86) || defined(YNN_COMPOSITES_NEON)
bool arch_is_blockwise_linear_supported(int32_t bits) noexcept;
size_t arch_packed_blockwise_linear_size(int32_t bits, size_t n, size_t k, size_t block_size, bool has_zero) noexcept;
void arch_pack_blockwise_linear_tiles(int32_t bits, size_t n, size_t k, size_t block_size, const void* weights,
                                      const void* scales, bool scales_fp16, const void* zeros, bool zeros_fp16,
                                      size_t tile_start, size_t tile_end, void* packed_tiles);
void arch_quantize_blockwise_blocks_f32(int32_t bits, size_t m, size_t k, size_t block_size, bool compute_bsum,
                                        const float* input, int8_t* quantized_input, float* input_scales,
                                        float* input_bsums, size_t block_start, size_t block_end);
void arch_blockwise_linear_f32(int32_t bits, size_t m, size_t n, size_t k, size_t block_size, bool has_zero,
                               float output_scale, const void* packed_tiles, const int8_t* quantized_input,
                               const float* input_scales, const float* input_bsums, float* output, size_t tile_start,
                               size_t tile_end);

bool arch_is_symmetric_linear_supported(int32_t bits) noexcept;
size_t arch_packed_symmetric_linear_size(int32_t bits, size_t n, size_t k, size_t block_size) noexcept;
void arch_pack_symmetric_linear_tiles(int32_t bits, size_t n, size_t k, size_t row_offset, size_t block_size,
                                      const void* weights, const void* scales, bool scales_fp16, size_t tile_start,
                                      size_t tile_end, void* packed_tiles);
bool arch_quantize_symmetric_blocks_f32(int32_t bits, size_t m, size_t k, size_t block_size, float static_scale,
                                        const float* input, int8_t* quantized_input, float* input_scales,
                                        size_t block_start, size_t block_end);
bool arch_symmetric_linear_f32(int32_t bits, size_t m, size_t n, size_t k, size_t block_size, float output_scale,
                               const void* packed_tiles, const int8_t* quantized_input, const float* input_scales,
                               float* output, size_t output_stride, size_t tile_start, size_t tile_end);
#else
inline bool arch_is_blockwise_linear_supported(int32_t) noexcept { return false; }
inline size_t arch_packed_blockwise_linear_size(int32_t, size_t, size_t, size_t, bool) noexcept { return 0; }
inline void arch_pack_blockwise_linear_tiles(int32_t, size_t, size_t, size_t, const void*, const void*, bool,
                                             const void*, bool, size_t, size_t, void*) {}
inline void arch_quantize_blockwise_blocks_f32(int32_t, size_t, size_t, size_t, bool, const float*, int8_t*, float*,
                                               float*, size_t, size_t) {}
inline void arch_blockwise_linear_f32(int32_t, size_t, size_t, size_t, size_t, bool, float, const void*, const int8_t*,
                                      const float*, const float*, float*, size_t, size_t) {}

inline bool arch_is_symmetric_linear_supported(int32_t) noexcept { return false; }
inline size_t arch_packed_symmetric_linear_size(int32_t, size_t, size_t, size_t) noexcept { return 0; }
inline void arch_pack_symmetric_linear_tiles(int32_t, size_t, size_t, size_t, size_t, const void*, const void*, bool,
                                             size_t, size_t, void*) {}
inline bool arch_quantize_symmetric_blocks_f32(int32_t, size_t, size_t, size_t, float, const float*, int8_t*, float*,
                                               size_t, size_t) {
    return false;
}
inline bool arch_symmetric_linear_f32(int32_t, size_t, size_t, size_t, size_t, float, const void*, const int8_t*,
                                      const float*, float*, size_t, size_t, size_t) {
    return false;
}
#endif

// --- hnsw_kernels variants (x86, neon) ---
template <bool BFloat>
inline float hnsw_weight_elem(const void* row, size_t column) noexcept {
    if constexpr (BFloat) {
        const uint32_t bits = static_cast<uint32_t>(static_cast<const uint16_t*>(row)[column]) << 16;
        float out = 0.0f;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    } else {
        return static_cast<const float*>(row)[column];
    }
}

inline float hnsw_read_meta(const uint8_t* base, bool f16, size_t idx) noexcept {
    if (f16) {
        uint16_t h = 0;
        std::memcpy(&h, base + idx * sizeof(uint16_t), sizeof(h));
        return static_cast<float>(half::from_bits(h));
    }
    float out = 0.0f;
    std::memcpy(&out, base + idx * sizeof(float), sizeof(out));
    return out;
}

#if defined(YNN_COMPOSITES_X86) || defined(YNN_COMPOSITES_NEON)
void arch_select_hnsw_dot_kernels(bool bf16, hnsw_dot_kernels& kernels) noexcept;
bool arch_hnsw_interleave_int4() noexcept;
bool arch_hnsw_quantized_slice_dot(const uint8_t* slice_ptr, size_t width, int32_t bits, int32_t group_size,
                                   bool scale_fp16, bool zero_fp16, bool has_zero, const int8_t* q_i8,
                                   const float* q_scale, const float* q_bsum, float* out_score) noexcept;
bool arch_hnsw_quantized_slices4_dot(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3,
                                     size_t width, int32_t bits, int32_t group_size, bool scale_fp16, bool zero_fp16,
                                     bool has_zero, const int8_t* q_i8, const float* q_scale, const float* q_bsum,
                                     float out[4]) noexcept;
#else
inline void arch_select_hnsw_dot_kernels(bool, hnsw_dot_kernels&) noexcept {}
inline bool arch_hnsw_interleave_int4() noexcept { return false; }
inline bool arch_hnsw_quantized_slice_dot(const uint8_t*, size_t, int32_t, int32_t, bool, bool, bool, const int8_t*,
                                          const float*, const float*, float*) noexcept {
    return false;
}
inline bool arch_hnsw_quantized_slices4_dot(const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, size_t,
                                            int32_t, int32_t, bool, bool, bool, const int8_t*, const float*,
                                            const float*, float[4]) noexcept {
    return false;
}
#endif

// --- embedding variants (x86, neon) ---
#if defined(YNN_COMPOSITES_X86) || defined(YNN_COMPOSITES_NEON)
size_t arch_quantize_symmetric_int8_f32(size_t count, float scale, const float* input, int8_t* output) noexcept;
size_t arch_argmax_chunk_f32(size_t count, const float* row_logits, size_t col_offset, float& best, int32_t& token,
                             bool& invalid) noexcept;
#else
inline size_t arch_quantize_symmetric_int8_f32(size_t, float, const float*, int8_t*) noexcept { return 0; }
inline size_t arch_argmax_chunk_f32(size_t, const float*, size_t, float&, int32_t&, bool&) noexcept { return 0; }
#endif

// --- normalization variants (x86, neon) ---
#if defined(YNN_COMPOSITES_X86) || defined(YNN_COMPOSITES_NEON)
bool arch_rms_norm_f32(size_t rows, size_t width, float epsilon, const float* input, const float* weight,
                       const float* residual, bool has_scale, float scale, float* output, size_t row_start,
                       size_t row_end) noexcept;
bool arch_rotary_f32(size_t sequence_rows, size_t heads, size_t width, bool shared_angles, const float* input,
                     const float* cosine, const float* sine, float* output, size_t row_start, size_t row_end) noexcept;
bool arch_rms_rotary_f32(size_t sequence_rows, size_t heads, size_t width, bool sliced, size_t input_rows,
                         size_t input_stride, size_t sequence_offset, size_t channel_offset, float epsilon,
                         bool shared_angles, const float* input, const float* weight, const float* cosine,
                         const float* sine, float* output, size_t vec_start, size_t vec_end) noexcept;
#else
inline bool arch_rms_norm_f32(size_t, size_t, float, const float*, const float*, const float*, bool, float, float*,
                              size_t, size_t) noexcept {
    return false;
}
inline bool arch_rotary_f32(size_t, size_t, size_t, bool, const float*, const float*, const float*, float*, size_t,
                            size_t) noexcept {
    return false;
}
inline bool arch_rms_rotary_f32(size_t, size_t, size_t, bool, size_t, size_t, size_t, size_t, float, bool, const float*,
                                const float*, const float*, const float*, float*, size_t, size_t) noexcept {
    return false;
}
#endif

// --- dense_linear variants (neon) ---
#if defined(YNN_COMPOSITES_NEON)
size_t arch_f32_to_bf16_slice(const float* src, uint16_t* dst, size_t count) noexcept;
size_t arch_bf16_to_f32_slice(const uint16_t* src, float* dst, size_t count) noexcept;
bool arch_quantize_dynamic_int8_rows_f32(size_t rows, size_t k, const float* input, int8_t* quantized, float* scales,
                                         int32_t* zero_points, size_t row_start, size_t row_end) noexcept;
size_t arch_dequantize_i32_row_f32(size_t n_valid, const int32_t* acc_row, const int32_t* wsum, float a_scale,
                                   int32_t a_zp, const float* ws, const float* bp, float* out_row) noexcept;
#else
inline size_t arch_f32_to_bf16_slice(const float*, uint16_t*, size_t) noexcept { return 0; }
inline size_t arch_bf16_to_f32_slice(const uint16_t*, float*, size_t) noexcept { return 0; }
inline bool arch_quantize_dynamic_int8_rows_f32(size_t, size_t, const float*, int8_t*, float*, int32_t*, size_t,
                                                size_t) noexcept {
    return false;
}
inline size_t arch_dequantize_i32_row_f32(size_t, const int32_t*, const int32_t*, float, int32_t, const float*,
                                          const float*, float*) noexcept {
    return 0;
}
#endif

// --- tensor_ops variants (neon) ---
#if defined(YNN_COMPOSITES_NEON)
bool arch_layer_norm_f32(size_t rows, size_t width, float epsilon, const float* input, const float* residual_in,
                         const float* weight, const float* bias, float* output_norm, float* output_residual,
                         size_t row_start, size_t row_end) noexcept;
bool arch_softmax_f32(size_t rows, size_t width, bool logarithmic, const float* input, float* output, size_t row_start,
                      size_t row_end) noexcept;
size_t arch_broadcast_binary_inner_f32(bool is_multiply, size_t chunk, size_t sa, size_t sb, const float* ap,
                                       const float* bp, float* op) noexcept;
#else
inline bool arch_layer_norm_f32(size_t, size_t, float, const float*, const float*, const float*, const float*, float*,
                                float*, size_t, size_t) noexcept {
    return false;
}
inline bool arch_softmax_f32(size_t, size_t, bool, const float*, float*, size_t, size_t) noexcept { return false; }
inline size_t arch_broadcast_binary_inner_f32(bool, size_t, size_t, size_t, const float*, const float*,
                                              float*) noexcept {
    return 0;
}
#endif

// --- attention_prefill & flash_attention variants (neon, accelerate) ---
#if defined(YNN_COMPOSITES_NEON)
bool arch_accumulate_grouped_value(size_t size, const int8_t* value, int32_t zero_point, float scale,
                                   const float* probabilities, size_t probability_stride, size_t heads, float* output,
                                   size_t output_stride) noexcept;
bool arch_score_grouped_key_token(size_t head_dim, size_t heads_per_group, const float* query_tile,
                                  const int8_t* key_token, const blockwise_quantization_f32_params& key_quantization,
                                  float* scores) noexcept;
bool arch_dot_i8_with_zp(size_t size, const int8_t* query, const int8_t* key, int32_t query_zero_point,
                         int32_t key_zero_point, int32_t* out_dot) noexcept;
size_t arch_scale_f32(size_t size, float scale, float* output) noexcept;
size_t arch_accumulate_i8_value_f32(size_t size, const int8_t* value, int32_t zero_point, float scale, float weight,
                                    float* output) noexcept;
bool arch_dot_f32_lanes8(const float* left, const float* right, size_t count, float* out_sum) noexcept;
size_t arch_accumulate_weighted_f32(float* dst, const float* src, float weight, size_t count) noexcept;
#else
inline bool arch_accumulate_grouped_value(size_t, const int8_t*, int32_t, float, const float*, size_t, size_t, float*,
                                          size_t) noexcept {
    return false;
}
inline bool arch_score_grouped_key_token(size_t, size_t, const float*, const int8_t*,
                                         const blockwise_quantization_f32_params&, float*) noexcept {
    return false;
}
inline bool arch_dot_i8_with_zp(size_t, const int8_t*, const int8_t*, int32_t, int32_t, int32_t*) noexcept {
    return false;
}
inline size_t arch_scale_f32(size_t, float, float*) noexcept { return 0; }
inline size_t arch_accumulate_i8_value_f32(size_t, const int8_t*, int32_t, float, float, float*) noexcept { return 0; }
inline bool arch_dot_f32_lanes8(const float*, const float*, size_t, float*) noexcept { return false; }
inline size_t arch_accumulate_weighted_f32(float*, const float*, float, size_t) noexcept { return 0; }
#endif

#if defined(YNN_COMPOSITES_ACCELERATE)
bool arch_sgemm_f32(bool transpose_b, size_t m, size_t n, size_t k, float alpha, const float* a, size_t lda,
                    const float* b, size_t ldb, float* c, size_t ldc) noexcept;
#else
inline bool arch_sgemm_f32(bool, size_t, size_t, size_t, float, const float*, size_t, const float*, size_t, float*,
                           size_t) noexcept {
    return false;
}
#endif

} // namespace ynn::detail

#endif // XNNPACK_YNNPACK_COMPOSITES_COMPOSITES_INTERNAL_H_
