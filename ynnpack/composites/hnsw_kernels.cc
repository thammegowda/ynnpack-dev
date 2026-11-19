// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ynnpack/composites/composites_internal.h"
#include "ynnpack/composites/hnsw.h"

namespace ynn {
namespace {

template <bool BFloat>
float dot_portable(const float* query, const void* row, size_t width) {
    std::array<float, 16> lanes{};
    size_t column = 0;
    for (; column + 16 <= width; column += 16) {
        for (size_t lane = 0; lane < 16; ++lane) {
            lanes[lane] =
                std::fma(query[column + lane], detail::hnsw_weight_elem<BFloat>(row, column + lane), lanes[lane]);
        }
    }
    for (size_t offset = 8; offset != 0; offset /= 2) {
        for (size_t lane = 0; lane < offset; ++lane) {
            lanes[lane] += lanes[lane + offset];
        }
    }
    float tail = 0.0f;
    for (; column < width; ++column) {
        tail = std::fma(query[column], detail::hnsw_weight_elem<BFloat>(row, column), tail);
    }
    return lanes[0] + tail;
}

template <bool BFloat>
void dot4_portable(const float* query, const void* r0, const void* r1, const void* r2, const void* r3, size_t width,
                   float out[4]) {
    out[0] = dot_portable<BFloat>(query, r0, width);
    out[1] = dot_portable<BFloat>(query, r1, width);
    out[2] = dot_portable<BFloat>(query, r2, width);
    out[3] = dot_portable<BFloat>(query, r3, width);
}

template <bool BFloat>
float dot_rows_portable(const void* left, const void* right, size_t width) {
    std::array<float, 16> lanes{};
    size_t column = 0;
    for (; column + 16 <= width; column += 16) {
        for (size_t lane = 0; lane < 16; ++lane) {
            lanes[lane] = std::fma(detail::hnsw_weight_elem<BFloat>(left, column + lane),
                                   detail::hnsw_weight_elem<BFloat>(right, column + lane), lanes[lane]);
        }
    }
    for (size_t offset = 8; offset != 0; offset /= 2) {
        for (size_t lane = 0; lane < offset; ++lane) {
            lanes[lane] += lanes[lane + offset];
        }
    }
    float tail = 0.0f;
    for (; column < width; ++column) {
        tail = std::fma(detail::hnsw_weight_elem<BFloat>(left, column), detail::hnsw_weight_elem<BFloat>(right, column),
                        tail);
    }
    return lanes[0] + tail;
}

template <bool BFloat>
void dot_rows4_portable(const void* query, const void* r0, const void* r1, const void* r2, const void* r3, size_t width,
                        float out[4]) {
    out[0] = dot_rows_portable<BFloat>(query, r0, width);
    out[1] = dot_rows_portable<BFloat>(query, r1, width);
    out[2] = dot_rows_portable<BFloat>(query, r2, width);
    out[3] = dot_rows_portable<BFloat>(query, r3, width);
}

} // namespace

hnsw_dot_kernels get_hnsw_dot_kernels(bool bf16) noexcept {
    hnsw_dot_kernels kernels;
    kernels.dot = bf16 ? dot_portable<true> : dot_portable<false>;
    kernels.dot4 = bf16 ? dot4_portable<true> : dot4_portable<false>;
    kernels.row_dot = bf16 ? dot_rows_portable<true> : dot_rows_portable<false>;
    kernels.row_dot4 = bf16 ? dot_rows4_portable<true> : dot_rows4_portable<false>;
    detail::arch_select_hnsw_dot_kernels(bf16, kernels);
    return kernels;
}

float hnsw_weight_element_f32(const void* row, size_t column, bool bf16) noexcept {
    return bf16 ? detail::hnsw_weight_elem<true>(row, column) : detail::hnsw_weight_elem<false>(row, column);
}

void hnsw_prefetch_slice(const void* slice_ptr, size_t slice_bytes) noexcept {
    const size_t limit = std::min<size_t>(slice_bytes, 1024);
    const auto* base = static_cast<const char*>(slice_ptr);
    for (size_t off = 0; off < limit; off += 128) {
        detail::prefetch_l1(base + off);
    }
}

size_t hnsw_quantized_slice_bytes(size_t width, int32_t bits, int32_t group_size, bool scale_fp16, bool zero_fp16,
                                  bool has_zero) noexcept {
    const size_t row_bytes = width / (8u / static_cast<size_t>(bits));
    const size_t B = width / static_cast<size_t>(group_size);
    const size_t scale_row_bytes = B * (scale_fp16 ? 2u : 4u);
    const size_t zero_row_bytes = has_zero ? B * (zero_fp16 ? 2u : 4u) : 0u;
    const size_t wsum_row_bytes = bits == 8 ? B * sizeof(int32_t) : 0u;
    const size_t raw = row_bytes + scale_row_bytes + zero_row_bytes + wsum_row_bytes;
    return raw >= 64 ? ((raw + 63u) & ~size_t{63}) : ((raw + 15u) & ~size_t{15});
}

void hnsw_pack_quantized_slice(uint8_t* dst, const void* w_row, const void* s_row, const void* z_row, size_t width,
                               int32_t bits, int32_t group_size, bool scale_fp16, bool zero_fp16,
                               bool has_zero) noexcept {
    const size_t row_bytes = width / (8u / static_cast<size_t>(bits));
    const size_t G = static_cast<size_t>(group_size);
    const size_t B = width / G;
    const size_t s_elem = scale_fp16 ? 2u : 4u;
    const size_t z_elem = zero_fp16 ? 2u : 4u;
    const auto* w_bytes = static_cast<const uint8_t*>(w_row);
    const auto* s_bytes = static_cast<const uint8_t*>(s_row);
    const auto* z_bytes = static_cast<const uint8_t*>(z_row);
    if (bits == 4 && B % 4 == 0 && detail::arch_hnsw_interleave_int4()) {
        const size_t chunks = G / 32;
        const size_t words_per_block = chunks * 4;
        uint8_t* ptr = dst;
        for (size_t b = 0; b < B; b += 4) {
            for (size_t k = 0; k < words_per_block; ++k) {
                for (size_t r = 0; r < 4; ++r) {
                    std::memcpy(ptr + (k * 4 + r) * 4, w_bytes + ((b + r) * words_per_block + k) * 4, 4);
                }
            }
            ptr += words_per_block * 16;
            std::memcpy(ptr, s_bytes + b * s_elem, 4 * s_elem);
            ptr += 4 * s_elem;
            if (has_zero) {
                std::memcpy(ptr, z_bytes + b * z_elem, 4 * z_elem);
                ptr += 4 * z_elem;
            }
        }
        return;
    }
    const size_t scale_row_bytes = B * s_elem;
    const size_t zero_row_bytes = has_zero ? B * z_elem : 0u;
    std::memcpy(dst, w_bytes, row_bytes);
    std::memcpy(dst + row_bytes, s_bytes, scale_row_bytes);
    if (has_zero) {
        std::memcpy(dst + row_bytes + scale_row_bytes, z_bytes, zero_row_bytes);
    }
    if (bits == 8) {
        const auto* w_i8 = static_cast<const int8_t*>(w_row);
        uint8_t* wsum_dst = dst + row_bytes + scale_row_bytes + zero_row_bytes;
        for (size_t b = 0; b < B; ++b) {
            int32_t wsum = 0;
            for (size_t c = 0; c < G; ++c) {
                wsum += static_cast<int32_t>(w_i8[b * G + c]);
            }
            std::memcpy(wsum_dst + b * sizeof(int32_t), &wsum, sizeof(int32_t));
        }
    }
}

void hnsw_prepare_quantized_query(const float* query, size_t width, int32_t bits, int32_t group_size, float input_scale,
                                  std::vector<int8_t>& q_i8, std::vector<float>& q_scale, std::vector<float>& q_bsum) {
    if (bits == 0) return;
    const size_t G = static_cast<size_t>(group_size);
    const size_t B = width / G;
    const bool interleave_i4 = bits == 4 && B % 4 == 0 && detail::arch_hnsw_interleave_int4();
    q_i8.resize(width);
    q_scale.resize(B);
    q_bsum.resize(B);
    for (size_t b = 0; b < B; ++b) {
        const float* x = query + b * G;
        if (bits == 8 && input_scale <= 0.0f) {
            float rmin = 0.0f, rmax = 0.0f;
            for (size_t i = 0; i < G; ++i) {
                rmin = std::min(rmin, x[i]);
                rmax = std::max(rmax, x[i]);
            }
            const float rdiff = rmax - rmin;
            float scale = 1.0f;
            float inv_scale = 0.0f;
            int32_t zp = -128;
            if (rdiff > 0.0f) {
                inv_scale = 255.0f / rdiff;
                scale = rdiff / 255.0f;
                const float min_error = -128.0f + rmin * inv_scale;
                const float max_error = 127.0f + rmax * inv_scale;
                const float fzp = min_error + max_error > 0.0f ? -128.0f - rmin * inv_scale : 127.0f - rmax * inv_scale;
                zp = static_cast<int32_t>(std::rint(std::clamp(fzp, -128.0f, 127.0f)));
            }
            q_scale[b] = scale;
            q_bsum[b] = static_cast<float>(zp);
            int8_t* dst = q_i8.data() + b * G;
            for (size_t i = 0; i < G; ++i) {
                const int32_t q = static_cast<int32_t>(std::round(x[i] * inv_scale)) + zp;
                dst[i] = static_cast<int8_t>(std::clamp<int32_t>(q, -128, 127));
            }
            continue;
        }
        float scale = input_scale;
        float inv_scale = 0.0f;
        if (scale > 0.0f) {
            inv_scale = 1.0f / scale;
        } else {
            float max_abs = 0.0f;
            for (size_t i = 0; i < G; ++i) max_abs = std::max(max_abs, std::abs(x[i]));
            scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
            inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 0.0f;
        }
        q_scale[b] = scale;
        const float min_q = input_scale > 0.0f ? -128.0f : -127.0f;
        if (bits == 8) {
            q_bsum[b] = 0.0f;
            int8_t* dst = q_i8.data() + b * G;
            for (size_t i = 0; i < G; ++i) {
                dst[i] = static_cast<int8_t>(std::clamp(std::nearbyint(x[i] * inv_scale), min_q, 127.0f));
            }
        } else if (bits == 4) {
            const size_t chunks = G / 32;
            int32_t bsum = 0;
            if (interleave_i4) {
                const size_t words_per_block = chunks * 4;
                const size_t base_k = (b / 4) * words_per_block;
                const size_t r = b & 3u;
                for (size_t c = 0; c < chunks; ++c) {
                    const float* xc = x + c * 32;
                    for (size_t i = 0; i < 16; ++i) {
                        const auto qe =
                            static_cast<int8_t>(std::clamp(std::nearbyint(xc[2 * i] * inv_scale), min_q, 127.0f));
                        const auto qo =
                            static_cast<int8_t>(std::clamp(std::nearbyint(xc[2 * i + 1] * inv_scale), min_q, 127.0f));
                        const size_t k = c * 4 + (i >> 2u);
                        const size_t lane = i & 3u;
                        int8_t* dst = q_i8.data() + (base_k + k) * 32 + r * 4 + lane;
                        dst[0] = qe;
                        dst[16] = qo;
                        bsum += static_cast<int32_t>(qe) + static_cast<int32_t>(qo);
                    }
                }
                q_bsum[b] = static_cast<float>(bsum) * scale;
                continue;
            }
            for (size_t c = 0; c < chunks; ++c) {
                const float* xc = x + c * 32;
                int8_t* dst = q_i8.data() + (b * chunks + c) * 32;
                for (size_t i = 0; i < 16; ++i) {
                    const auto qe =
                        static_cast<int8_t>(std::clamp(std::nearbyint(xc[2 * i] * inv_scale), min_q, 127.0f));
                    const auto qo =
                        static_cast<int8_t>(std::clamp(std::nearbyint(xc[2 * i + 1] * inv_scale), min_q, 127.0f));
                    dst[i] = qe;
                    dst[16 + i] = qo;
                    bsum += static_cast<int32_t>(qe) + static_cast<int32_t>(qo);
                }
            }
            q_bsum[b] = static_cast<float>(bsum) * scale;
        } else if (bits == 2) {
            q_bsum[b] = 0.0f;
            const size_t quarter = G / 4;
            int8_t* dst = q_i8.data() + b * G;
            for (size_t i = 0; i < quarter; ++i) {
                dst[0 * quarter + i] =
                    static_cast<int8_t>(std::clamp(std::nearbyint(x[4 * i + 0] * inv_scale), min_q, 127.0f));
                dst[1 * quarter + i] =
                    static_cast<int8_t>(std::clamp(std::nearbyint(x[4 * i + 1] * inv_scale), min_q, 127.0f));
                dst[2 * quarter + i] =
                    static_cast<int8_t>(std::clamp(std::nearbyint(x[4 * i + 2] * inv_scale), min_q, 127.0f));
                dst[3 * quarter + i] =
                    static_cast<int8_t>(std::clamp(std::nearbyint(x[4 * i + 3] * inv_scale), min_q, 127.0f));
            }
        }
    }
}

float hnsw_quantized_slice_dot(const uint8_t* slice_ptr, size_t width, int32_t bits, int32_t group_size,
                               bool scale_fp16, bool zero_fp16, bool has_zero, const int8_t* q_i8, const float* q_scale,
                               const float* q_bsum) noexcept {
    float fast_score = 0.0f;
    if (detail::arch_hnsw_quantized_slice_dot(slice_ptr, width, bits, group_size, scale_fp16, zero_fp16, has_zero, q_i8,
                                              q_scale, q_bsum, &fast_score)) {
        return fast_score;
    }

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
            int32_t dot_i32 = 0;
            for (size_t c = 0; c < G; ++c) {
                dot_i32 += static_cast<int32_t>(q_ptr[c]) * static_cast<int32_t>(w_ptr[c]);
            }
            const int32_t zp = static_cast<int32_t>(q_bsum[b]);
            if (zp != 0) {
                int32_t wsum = 0;
                std::memcpy(&wsum, wsum_ptr + b * sizeof(int32_t), sizeof(int32_t));
                dot_i32 -= zp * wsum;
            }
            score += static_cast<float>(dot_i32) * (q_scale[b] * detail::hnsw_read_meta(scale_ptr, scale_fp16, b));
        }
    } else if (bits == 4) {
        const size_t chunks = G / 32;
        for (size_t b = 0; b < B; ++b) {
            int32_t dot_i32 = 0;
            for (size_t c = 0; c < chunks; ++c) {
                const uint8_t* wp = slice_ptr + (b * chunks + c) * 16;
                const int8_t* qp = q_i8 + (b * chunks + c) * 32;
                for (size_t i = 0; i < 16; ++i) {
                    const uint8_t byte = wp[i];
                    const auto we = static_cast<int8_t>(static_cast<uint8_t>(byte << 4)) >> 4;
                    const auto wo = static_cast<int8_t>(byte) >> 4;
                    dot_i32 += static_cast<int32_t>(we) * qp[i] + static_cast<int32_t>(wo) * qp[16 + i];
                }
            }
            score += static_cast<float>(dot_i32) * (q_scale[b] * detail::hnsw_read_meta(scale_ptr, scale_fp16, b));
            if (has_zero) {
                score += q_bsum[b] * detail::hnsw_read_meta(zero_ptr, zero_fp16, b);
            }
        }
    } else if (bits == 2) {
        const size_t quarter = G / 4;
        for (size_t b = 0; b < B; ++b) {
            const uint8_t* wp = slice_ptr + b * quarter;
            const int8_t* qp = q_i8 + b * G;
            int32_t dot_i32 = 0;
            for (size_t c = 0; c < quarter; ++c) {
                const auto raw = static_cast<int8_t>(wp[c]);
                dot_i32 += static_cast<int32_t>(static_cast<int8_t>(raw << 6) >> 6) * qp[0 * quarter + c] +
                           static_cast<int32_t>(static_cast<int8_t>(raw << 4) >> 6) * qp[1 * quarter + c] +
                           static_cast<int32_t>(static_cast<int8_t>(raw << 2) >> 6) * qp[2 * quarter + c] +
                           static_cast<int32_t>(raw >> 6) * qp[3 * quarter + c];
            }
            score += static_cast<float>(dot_i32) * (q_scale[b] * detail::hnsw_read_meta(scale_ptr, scale_fp16, b));
        }
    }
    return score;
}

void hnsw_quantized_slices4_dot(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3,
                                size_t width, int32_t bits, int32_t group_size, bool scale_fp16, bool zero_fp16,
                                bool has_zero, const int8_t* q_i8, const float* q_scale, const float* q_bsum,
                                float out[4]) noexcept {
    if (detail::arch_hnsw_quantized_slices4_dot(s0, s1, s2, s3, width, bits, group_size, scale_fp16, zero_fp16,
                                                has_zero, q_i8, q_scale, q_bsum, out)) {
        return;
    }
    out[0] =
        hnsw_quantized_slice_dot(s0, width, bits, group_size, scale_fp16, zero_fp16, has_zero, q_i8, q_scale, q_bsum);
    out[1] =
        hnsw_quantized_slice_dot(s1, width, bits, group_size, scale_fp16, zero_fp16, has_zero, q_i8, q_scale, q_bsum);
    out[2] =
        hnsw_quantized_slice_dot(s2, width, bits, group_size, scale_fp16, zero_fp16, has_zero, q_i8, q_scale, q_bsum);
    out[3] =
        hnsw_quantized_slice_dot(s3, width, bits, group_size, scale_fp16, zero_fp16, has_zero, q_i8, q_scale, q_bsum);
}

} // namespace ynn
