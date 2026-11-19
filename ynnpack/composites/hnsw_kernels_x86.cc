// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ynnpack/base/arch.h"
#include "ynnpack/composites/composites_internal.h"

namespace ynn::detail {
namespace {

inline bool has_x86_vnni() noexcept { return is_arch_supported(arch_flag::avx512 | arch_flag::avx512vnni); }

template <bool BFloat>
YNN_X86_TARGET_AVX512 inline __m512 load_avx512_row(const void* row, size_t col) noexcept {
    if constexpr (BFloat) {
        const auto* bytes = static_cast<const uint8_t*>(row);
        const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bytes + col * sizeof(uint16_t)));
        return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
    } else {
        return _mm512_loadu_ps(static_cast<const float*>(row) + col);
    }
}

template <bool BFloat>
YNN_X86_TARGET_AVX512 float dot_avx512(const float* query, const void* row, size_t width) {
    __m512 sum = _mm512_setzero_ps();
    size_t column = 0;
    for (; column + 16 <= width; column += 16) {
        sum = _mm512_fmadd_ps(_mm512_loadu_ps(query + column), load_avx512_row<BFloat>(row, column), sum);
    }
    float tail = 0.0f;
    for (; column < width; ++column) {
        tail = std::fma(query[column], hnsw_weight_elem<BFloat>(row, column), tail);
    }
    return _mm512_reduce_add_ps(sum) + tail;
}

template <bool BFloat>
YNN_X86_TARGET_AVX512 void dot4_avx512(const float* query, const void* r0, const void* r1, const void* r2,
                                       const void* r3, size_t width, float out[4]) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    size_t column = 0;
    for (; column + 16 <= width; column += 16) {
        const __m512 q = _mm512_loadu_ps(query + column);
        s0 = _mm512_fmadd_ps(q, load_avx512_row<BFloat>(r0, column), s0);
        s1 = _mm512_fmadd_ps(q, load_avx512_row<BFloat>(r1, column), s1);
        s2 = _mm512_fmadd_ps(q, load_avx512_row<BFloat>(r2, column), s2);
        s3 = _mm512_fmadd_ps(q, load_avx512_row<BFloat>(r3, column), s3);
    }
    out[0] = _mm512_reduce_add_ps(s0);
    out[1] = _mm512_reduce_add_ps(s1);
    out[2] = _mm512_reduce_add_ps(s2);
    out[3] = _mm512_reduce_add_ps(s3);
    for (; column < width; ++column) {
        const float q = query[column];
        out[0] = std::fma(q, hnsw_weight_elem<BFloat>(r0, column), out[0]);
        out[1] = std::fma(q, hnsw_weight_elem<BFloat>(r1, column), out[1]);
        out[2] = std::fma(q, hnsw_weight_elem<BFloat>(r2, column), out[2]);
        out[3] = std::fma(q, hnsw_weight_elem<BFloat>(r3, column), out[3]);
    }
}

template <bool BFloat>
YNN_X86_TARGET_AVX512 float dot_rows_avx512(const void* left, const void* right, size_t width) {
    __m512 sum = _mm512_setzero_ps();
    size_t column = 0;
    for (; column + 16 <= width; column += 16) {
        sum = _mm512_fmadd_ps(load_avx512_row<BFloat>(left, column), load_avx512_row<BFloat>(right, column), sum);
    }
    float tail = 0.0f;
    for (; column < width; ++column) {
        tail = std::fma(hnsw_weight_elem<BFloat>(left, column), hnsw_weight_elem<BFloat>(right, column), tail);
    }
    return _mm512_reduce_add_ps(sum) + tail;
}

template <bool BFloat>
YNN_X86_TARGET_AVX512 void dot_rows4_avx512(const void* query, const void* r0, const void* r1, const void* r2,
                                            const void* r3, size_t width, float out[4]) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    size_t column = 0;
    for (; column + 16 <= width; column += 16) {
        const __m512 q = load_avx512_row<BFloat>(query, column);
        s0 = _mm512_fmadd_ps(q, load_avx512_row<BFloat>(r0, column), s0);
        s1 = _mm512_fmadd_ps(q, load_avx512_row<BFloat>(r1, column), s1);
        s2 = _mm512_fmadd_ps(q, load_avx512_row<BFloat>(r2, column), s2);
        s3 = _mm512_fmadd_ps(q, load_avx512_row<BFloat>(r3, column), s3);
    }
    out[0] = _mm512_reduce_add_ps(s0);
    out[1] = _mm512_reduce_add_ps(s1);
    out[2] = _mm512_reduce_add_ps(s2);
    out[3] = _mm512_reduce_add_ps(s3);
    for (; column < width; ++column) {
        const float q = hnsw_weight_elem<BFloat>(query, column);
        out[0] = std::fma(q, hnsw_weight_elem<BFloat>(r0, column), out[0]);
        out[1] = std::fma(q, hnsw_weight_elem<BFloat>(r1, column), out[1]);
        out[2] = std::fma(q, hnsw_weight_elem<BFloat>(r2, column), out[2]);
        out[3] = std::fma(q, hnsw_weight_elem<BFloat>(r3, column), out[3]);
    }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"))) int32_t
dot_i8_block_vnni(const int8_t* q_ptr, const int8_t* w_ptr, size_t G, int32_t wsum) noexcept {
    const __m512i sign_flip = _mm512_set1_epi8(-128);
    __m512i acc = _mm512_setzero_si512();
    size_t c = 0;
    for (; c + 64 <= G; c += 64) {
        const __m512i qu = _mm512_xor_si512(_mm512_loadu_si512(q_ptr + c), sign_flip);
        acc = _mm512_dpbusd_epi32(acc, qu, _mm512_loadu_si512(w_ptr + c));
    }
    int32_t dot_i32 = _mm512_reduce_add_epi32(acc);
    int32_t tail_wsum = 0;
    for (; c < G; ++c) {
        dot_i32 += static_cast<int32_t>(q_ptr[c]) * static_cast<int32_t>(w_ptr[c]);
        tail_wsum += static_cast<int32_t>(w_ptr[c]);
    }
    return dot_i32 - 128 * (wsum - tail_wsum);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"))) void dot4_i8_slices_vnni(
    const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3, size_t B, size_t G, size_t row_bytes,
    bool scale_fp16, bool zero_fp16, bool has_zero, const int8_t* q_i8, const float* q_scale, const float* q_bsum,
    float out[4]) noexcept {
    const auto* w0 = reinterpret_cast<const int8_t*>(s0);
    const auto* w1 = reinterpret_cast<const int8_t*>(s1);
    const auto* w2 = reinterpret_cast<const int8_t*>(s2);
    const auto* w3 = reinterpret_cast<const int8_t*>(s3);
    const uint8_t* sc0 = s0 + row_bytes;
    const uint8_t* sc1 = s1 + row_bytes;
    const uint8_t* sc2 = s2 + row_bytes;
    const uint8_t* sc3 = s3 + row_bytes;
    const size_t wsum_off = row_bytes + B * (scale_fp16 ? 2u : 4u) + (has_zero ? B * (zero_fp16 ? 2u : 4u) : 0u);
    const __m512i sign_flip = _mm512_set1_epi8(-128);
    out[0] = out[1] = out[2] = out[3] = 0.0f;
    for (size_t b = 0; b < B; ++b) {
        const int8_t* q_ptr = q_i8 + b * G;
        __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
        __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
        size_t c = 0;
        for (; c + 64 <= G; c += 64) {
            const __m512i qu = _mm512_xor_si512(_mm512_loadu_si512(q_ptr + c), sign_flip);
            a0 = _mm512_dpbusd_epi32(a0, qu, _mm512_loadu_si512(w0 + b * G + c));
            a1 = _mm512_dpbusd_epi32(a1, qu, _mm512_loadu_si512(w1 + b * G + c));
            a2 = _mm512_dpbusd_epi32(a2, qu, _mm512_loadu_si512(w2 + b * G + c));
            a3 = _mm512_dpbusd_epi32(a3, qu, _mm512_loadu_si512(w3 + b * G + c));
        }
        int32_t d0 = _mm512_reduce_add_epi32(a0);
        int32_t d1 = _mm512_reduce_add_epi32(a1);
        int32_t d2 = _mm512_reduce_add_epi32(a2);
        int32_t d3 = _mm512_reduce_add_epi32(a3);
        int32_t tw0 = 0, tw1 = 0, tw2 = 0, tw3 = 0;
        for (; c < G; ++c) {
            const int32_t qv = static_cast<int32_t>(q_ptr[c]);
            d0 += qv * w0[b * G + c];
            d1 += qv * w1[b * G + c];
            d2 += qv * w2[b * G + c];
            d3 += qv * w3[b * G + c];
            tw0 += w0[b * G + c];
            tw1 += w1[b * G + c];
            tw2 += w2[b * G + c];
            tw3 += w3[b * G + c];
        }
        int32_t ws0 = 0, ws1 = 0, ws2 = 0, ws3 = 0;
        std::memcpy(&ws0, s0 + wsum_off + b * sizeof(int32_t), sizeof(int32_t));
        std::memcpy(&ws1, s1 + wsum_off + b * sizeof(int32_t), sizeof(int32_t));
        std::memcpy(&ws2, s2 + wsum_off + b * sizeof(int32_t), sizeof(int32_t));
        std::memcpy(&ws3, s3 + wsum_off + b * sizeof(int32_t), sizeof(int32_t));
        const int32_t zp = static_cast<int32_t>(q_bsum[b]);
        d0 -= (zp + 128) * ws0 - 128 * tw0;
        d1 -= (zp + 128) * ws1 - 128 * tw1;
        d2 -= (zp + 128) * ws2 - 128 * tw2;
        d3 -= (zp + 128) * ws3 - 128 * tw3;
        out[0] += static_cast<float>(d0) * (q_scale[b] * hnsw_read_meta(sc0, scale_fp16, b));
        out[1] += static_cast<float>(d1) * (q_scale[b] * hnsw_read_meta(sc1, scale_fp16, b));
        out[2] += static_cast<float>(d2) * (q_scale[b] * hnsw_read_meta(sc2, scale_fp16, b));
        out[3] += static_cast<float>(d3) * (q_scale[b] * hnsw_read_meta(sc3, scale_fp16, b));
    }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"))) int32_t
dot_i4_block_vnni(const uint8_t* w_block, const int8_t* q_block, size_t chunks) noexcept {
    const __m128i sign_flip = _mm_set1_epi8(-128);
    const __m128i ones = _mm_set1_epi8(1);
    const __m128i low_mask = _mm_set1_epi8(0x0F);
    const __m128i lut = _mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m128i acc = _mm_setzero_si128();
    __m128i wsum = _mm_setzero_si128();
    for (size_t c = 0; c < chunks; ++c) {
        const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_block + c * 16));
        const __m128i we = _mm_shuffle_epi8(lut, _mm_and_si128(raw, low_mask));
        const __m128i wo = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(raw, 4), low_mask));
        const __m128i qe =
            _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q_block + c * 32)), sign_flip);
        const __m128i qo =
            _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q_block + c * 32 + 16)), sign_flip);
        acc = _mm_dpbusd_epi32(_mm_dpbusd_epi32(acc, qe, we), qo, wo);
        wsum = _mm_dpbusd_epi32(_mm_dpbusd_epi32(wsum, ones, we), ones, wo);
    }
    const __m128i corr = _mm_sub_epi32(acc, _mm_slli_epi32(wsum, 7));
    const __m128i sum64 = _mm_add_epi32(corr, _mm_srli_si128(corr, 8));
    const __m128i sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
    return _mm_cvtsi128_si32(sum32);
}

} // namespace

void arch_select_hnsw_dot_kernels(bool bf16, hnsw_dot_kernels& kernels) noexcept {
    if (is_arch_supported(arch_flag::avx512f | arch_flag::avx512bw | arch_flag::fma3)) {
        kernels.dot = bf16 ? dot_avx512<true> : dot_avx512<false>;
        kernels.dot4 = bf16 ? dot4_avx512<true> : dot4_avx512<false>;
        kernels.row_dot = bf16 ? dot_rows_avx512<true> : dot_rows_avx512<false>;
        kernels.row_dot4 = bf16 ? dot_rows4_avx512<true> : dot_rows4_avx512<false>;
    }
}

bool arch_hnsw_interleave_int4() noexcept { return false; }

bool arch_hnsw_quantized_slice_dot(const uint8_t* slice_ptr, size_t width, int32_t bits, int32_t group_size,
                                   bool scale_fp16, bool zero_fp16, bool has_zero, const int8_t* q_i8,
                                   const float* q_scale, const float* q_bsum, float* out_score) noexcept {
    if (!has_x86_vnni() || (bits != 8 && bits != 4)) return false;
    const size_t G = static_cast<size_t>(group_size);
    const size_t B = width / G;
    const size_t row_bytes = width / (8u / static_cast<size_t>(bits));
    const size_t scale_row_bytes = B * (scale_fp16 ? 2u : 4u);
    const uint8_t* scale_ptr = slice_ptr + row_bytes;
    const uint8_t* zero_ptr = scale_ptr + scale_row_bytes;
    float score = 0.0f;
    if (bits == 8) {
        const size_t zero_row_bytes = has_zero ? B * (zero_fp16 ? 2u : 4u) : 0u;
        const uint8_t* wsum_ptr = zero_ptr + zero_row_bytes;
        for (size_t b = 0; b < B; ++b) {
            const int8_t* q_ptr = q_i8 + b * G;
            const auto* w_ptr = reinterpret_cast<const int8_t*>(slice_ptr + b * G);
            int32_t wsum = 0;
            std::memcpy(&wsum, wsum_ptr + b * sizeof(int32_t), sizeof(int32_t));
            int32_t dot_i32 = dot_i8_block_vnni(q_ptr, w_ptr, G, wsum);
            const int32_t zp = static_cast<int32_t>(q_bsum[b]);
            if (zp != 0) dot_i32 -= zp * wsum;
            score += static_cast<float>(dot_i32) * (q_scale[b] * hnsw_read_meta(scale_ptr, scale_fp16, b));
        }
    } else {
        const size_t chunks = G / 32;
        for (size_t b = 0; b < B; ++b) {
            const int32_t dot_i32 = dot_i4_block_vnni(slice_ptr + b * chunks * 16, q_i8 + b * chunks * 32, chunks);
            score += static_cast<float>(dot_i32) * (q_scale[b] * hnsw_read_meta(scale_ptr, scale_fp16, b));
            if (has_zero) score += q_bsum[b] * hnsw_read_meta(zero_ptr, zero_fp16, b);
        }
    }
    *out_score = score;
    return true;
}

bool arch_hnsw_quantized_slices4_dot(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3,
                                     size_t width, int32_t bits, int32_t group_size, bool scale_fp16, bool zero_fp16,
                                     bool has_zero, const int8_t* q_i8, const float* q_scale, const float* q_bsum,
                                     float out[4]) noexcept {
    if (bits == 8 && has_x86_vnni()) {
        const size_t G = static_cast<size_t>(group_size);
        const size_t B = width / G;
        dot4_i8_slices_vnni(s0, s1, s2, s3, B, G, width, scale_fp16, zero_fp16, has_zero, q_i8, q_scale, q_bsum, out);
        return true;
    }
    return false;
}

} // namespace ynn::detail
