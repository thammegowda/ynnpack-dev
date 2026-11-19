// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "ynnpack/base/half.h"
#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/composites_internal.h"

namespace ynn {
namespace {

constexpr size_t NR_TILE = 32;

inline float calibrate_scalar_f32(float val, float output_scale) noexcept {
    if (output_scale <= 0.0f) return val;
    const float q = std::clamp(std::nearbyint(val / output_scale), -128.0f, 127.0f);
    return q * output_scale;
}

} // namespace

bool is_blockwise_linear_supported(int32_t bits) { return detail::arch_is_blockwise_linear_supported(bits); }

size_t blockwise_linear_tile_count(size_t n) { return (n + NR_TILE - 1) / NR_TILE; }

size_t packed_blockwise_linear_size(int32_t bits, size_t n, size_t k, size_t block_size, bool has_zero) {
    if (!is_blockwise_linear_supported(bits) || block_size == 0 || k == 0 || n == 0) return 0;
    return detail::arch_packed_blockwise_linear_size(bits, n, k, block_size, has_zero);
}

void pack_blockwise_linear_tiles(int32_t bits, size_t n, size_t k, size_t block_size, const void* weights,
                                 const void* scales, bool scales_fp16, const void* zeros, bool zeros_fp16,
                                 size_t tile_start, size_t tile_end, void* packed_tiles) {
    if (!is_blockwise_linear_supported(bits) || block_size == 0) return;
    detail::arch_pack_blockwise_linear_tiles(bits, n, k, block_size, weights, scales, scales_fp16, zeros, zeros_fp16,
                                             tile_start, tile_end, packed_tiles);
}

void quantize_blockwise_blocks_f32(int32_t bits, size_t m, size_t k, size_t block_size, bool compute_bsum,
                                   const float* input, int8_t* quantized_input, float* input_scales, float* input_bsums,
                                   size_t block_start, size_t block_end) {
    if (!is_blockwise_linear_supported(bits) || block_size == 0) return;
    detail::arch_quantize_blockwise_blocks_f32(bits, m, k, block_size, compute_bsum, input, quantized_input,
                                               input_scales, input_bsums, block_start, block_end);
}

void quantize_blockwise_activations_f32(int32_t bits, size_t m, size_t k, size_t block_size, bool compute_bsum,
                                        const float* input, int8_t* quantized_input, float* input_scales,
                                        float* input_bsums, size_t row_start, size_t row_end) {
    if (block_size == 0) return;
    const size_t B = k / block_size;
    row_end = std::min(row_end, m);
    if (row_start >= row_end) return;
    quantize_blockwise_blocks_f32(bits, m, k, block_size, compute_bsum, input, quantized_input, input_scales,
                                  input_bsums, row_start * B, row_end * B);
}

void blockwise_linear_f32(int32_t bits, size_t m, size_t n, size_t k, size_t block_size, bool has_zero,
                          float output_scale, const void* packed_tiles, const int8_t* quantized_input,
                          const float* input_scales, const float* input_bsums, float* output, size_t tile_start,
                          size_t tile_end) {
    if (!is_blockwise_linear_supported(bits) || block_size == 0 || m == 0) return;
    detail::arch_blockwise_linear_f32(bits, m, n, k, block_size, has_zero, output_scale, packed_tiles, quantized_input,
                                      input_scales, input_bsums, output, tile_start, tile_end);
}

bool is_symmetric_linear_supported(int32_t bits) { return detail::arch_is_symmetric_linear_supported(bits); }

size_t symmetric_linear_padded_block(size_t block_size) { return ((block_size + 31) / 32) * 32; }

size_t packed_symmetric_linear_size(int32_t bits, size_t n, size_t k, size_t block_size) {
    if (!is_symmetric_linear_supported(bits) || block_size == 0 || k == 0 || n == 0) return 0;
    return detail::arch_packed_symmetric_linear_size(bits, n, k, block_size);
}

void pack_symmetric_linear_tiles(int32_t bits, size_t n, size_t k, size_t row_offset, size_t block_size,
                                 const void* weights, const void* scales, bool scales_fp16, size_t tile_start,
                                 size_t tile_end, void* packed_tiles) {
    if (!is_symmetric_linear_supported(bits) || block_size == 0) return;
    detail::arch_pack_symmetric_linear_tiles(bits, n, k, row_offset, block_size, weights, scales, scales_fp16,
                                             tile_start, tile_end, packed_tiles);
}

void quantize_symmetric_blocks_f32(int32_t bits, size_t m, size_t k, size_t block_size, float static_scale,
                                   const float* input, int8_t* quantized_input, float* input_scales, size_t block_start,
                                   size_t block_end) {
    const size_t B = k / block_size;
    const size_t total_blocks = m * B;
    block_end = std::min(block_end, total_blocks);
    if (block_start >= block_end) return;
    if (detail::arch_quantize_symmetric_blocks_f32(bits, m, k, block_size, static_scale, input, quantized_input,
                                                   input_scales, block_start, block_end)) {
        return;
    }

    const size_t padded_block = symmetric_linear_padded_block(block_size);
    for (size_t fb = block_start; fb < block_end; ++fb) {
        const size_t row = fb / B;
        const size_t b = fb % B;
        const float* x_block = input + row * k + b * block_size;
        float scale = static_scale;
        float inv_scale = 0.0f;
        if (static_scale <= 0.0f) {
            float amax = 0.0f;
            for (size_t col = 0; col < block_size; ++col) amax = std::max(amax, std::abs(x_block[col]));
            scale = amax / 127.0f;
            inv_scale = amax > 0.0f ? 127.0f / amax : 0.0f;
        }
        input_scales[fb] = scale;
        int8_t* qp = quantized_input + fb * padded_block;
        std::memset(qp, 0, padded_block);
        for (size_t col = 0; col < block_size; ++col) {
            const float v = static_scale > 0.0f ? (x_block[col] / scale) : (x_block[col] * inv_scale);
            qp[col] = static_cast<int8_t>(std::clamp(std::nearbyint(v), -128.0f, 127.0f));
        }
    }
}

void symmetric_linear_f32(int32_t bits, size_t m, size_t n, size_t k, size_t block_size, float output_scale,
                          const void* packed_tiles, const int8_t* quantized_input, const float* input_scales,
                          float* output, size_t output_stride, size_t tile_start, size_t tile_end) {
    if (!is_symmetric_linear_supported(bits) || block_size == 0) return;
    const size_t tiles = blockwise_linear_tile_count(n);
    tile_end = std::min(tile_end, tiles);
    if (tile_start >= tile_end || m == 0) return;
    if (detail::arch_symmetric_linear_f32(bits, m, n, k, block_size, output_scale, packed_tiles, quantized_input,
                                          input_scales, output, output_stride, tile_start, tile_end)) {
        return;
    }

    const size_t B = k / block_size;
    const size_t padded_block = symmetric_linear_padded_block(block_size);
    const size_t chunks = padded_block / 32;
    const size_t steps_per_chunk = static_cast<size_t>(bits);
    const size_t vals_per_byte = static_cast<size_t>(8 / bits);
    const size_t weight_bytes = chunks * steps_per_chunk * 128;
    const size_t block_bytes = weight_bytes + 128;
    const size_t tile_bytes = B * block_bytes;
    const uint8_t mask = static_cast<uint8_t>((1 << bits) - 1);
    const int32_t sign_bit = 1 << (bits - 1);
    const int32_t full_range = 1 << bits;
    const auto* base = static_cast<const uint8_t*>(packed_tiles);

    for (size_t tile = tile_start; tile < tile_end; ++tile) {
        const size_t n0 = tile * 32;
        const size_t n_valid = std::min<size_t>(32, n - n0);
        const size_t col_off = (output_stride == 32 && tile_end == tile_start + 1) ? 0 : n0;
        const uint8_t* tile_ptr = base + tile * tile_bytes;

        for (size_t row_m = 0; row_m < m; ++row_m) {
            float acc[32] = {};
            const uint8_t* p_block = tile_ptr;
            for (size_t b = 0; b < B; ++b, p_block += block_bytes) {
                int32_t d[32] = {};
                const int8_t* qp = quantized_input + (row_m * B + b) * padded_block;
                const uint8_t* p = p_block;
                for (size_t c = 0; c < chunks; ++c) {
                    for (size_t step = 0; step < steps_per_chunk; ++step) {
                        const size_t col_base = c * 32 + step * (32 / steps_per_chunk);
                        for (size_t r = 0; r < n_valid; ++r) {
                            uint32_t word = 0;
                            std::memcpy(&word, p + r * 4, sizeof(uint32_t));
                            for (size_t bi = 0; bi < 4; ++bi) {
                                const uint8_t byte = static_cast<uint8_t>((word >> (bi * 8)) & 0xFFu);
                                for (size_t v = 0; v < vals_per_byte; ++v) {
                                    int32_t code = static_cast<int32_t>((byte >> (v * bits)) & mask);
                                    if (code & sign_bit) code -= full_range;
                                    d[r] += code * static_cast<int32_t>(qp[col_base + bi * vals_per_byte + v]);
                                }
                            }
                        }
                        p += 128;
                    }
                }
                const auto* sp = reinterpret_cast<const float*>(p_block + weight_bytes);
                const float as = input_scales[row_m * B + b];
                for (size_t r = 0; r < n_valid; ++r) acc[r] += static_cast<float>(d[r]) * (sp[r] * as);
            }
            float* dst = output + row_m * output_stride + col_off;
            for (size_t r = 0; r < n_valid; ++r) dst[r] = calibrate_scalar_f32(acc[r], output_scale);
        }
    }
}

void portable_packed_linear_unpack_weights(int32_t bits, const uint8_t* raw, size_t n, size_t k, size_t row_bytes,
                                           int8_t* out) {
    const size_t vpb = static_cast<size_t>(8 / bits);
    const int shift = 32 - bits;
    const uint32_t mask = (1u << bits) - 1u;
    for (size_t r = 0; r < n; ++r) {
        for (size_t c = 0; c < k; ++c) {
            const uint8_t b = raw[r * row_bytes + c / vpb];
            const uint32_t u = static_cast<uint32_t>((b >> ((c % vpb) * bits)) & mask);
            out[r * k + c] = static_cast<int8_t>(static_cast<int32_t>(u << shift) >> shift);
        }
    }
}

void portable_packed_linear_decode_meta(const void* meta, bool is_fp16, size_t count, float* out) {
    if (!is_fp16) {
        std::memcpy(out, meta, count * sizeof(float));
        return;
    }
    const auto* s = static_cast<const uint16_t*>(meta);
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<float>(half::from_bits(s[i]));
    }
}

void portable_packed_linear_f32(size_t rows, size_t n, size_t k, size_t block_size, float input_scale,
                                float output_scale, bool has_zero, const float* x, const int8_t* w, const float* scales,
                                const float* zeros, float* y, size_t row_start, size_t row_end) {
    row_end = std::min(row_end, rows);
    if (row_start >= row_end || n == 0 || k == 0 || block_size == 0) return;
    const size_t blocks = k / block_size;
    std::vector<int8_t> qx(k);
    std::vector<float> act_bsum(has_zero ? blocks : 0, 0.0f);
    for (size_t m = row_start; m < row_end; ++m) {
        const float* xm = x + m * k;
        float a_scale = input_scale;
        int32_t a_zp = 0;
        if (input_scale > 0.0f) {
            const float inv = 1.0f / input_scale;
            for (size_t c = 0; c < k; ++c)
                qx[c] = static_cast<int8_t>(std::clamp(std::nearbyint(xm[c] * inv), -128.0f, 127.0f));
        } else {
            quantize_dynamic_int8_rows_f32(1, k, xm, qx.data(), &a_scale, &a_zp, 0, 1);
        }
        if (has_zero) {
            for (size_t b = 0; b < blocks; ++b) {
                const float* xb = xm + b * block_size;
                float bsum = 0.0f;
                for (size_t c = 0; c < block_size; ++c) bsum += xb[c];
                act_bsum[b] = bsum;
            }
        }
        for (size_t r = 0; r < n; ++r) {
            float acc = 0.0f;
            for (size_t b = 0; b < blocks; ++b) {
                int32_t dot = 0;
                for (size_t c = 0; c < block_size; ++c) {
                    const size_t idx = b * block_size + c;
                    dot += (static_cast<int32_t>(qx[idx]) - a_zp) * static_cast<int32_t>(w[r * k + idx]);
                }
                acc += static_cast<float>(dot) * a_scale * scales[r * blocks + b];
                if (has_zero) acc += act_bsum[b] * zeros[r * blocks + b];
            }
            y[m * n + r] = calibrate_scalar_f32(acc, output_scale);
        }
    }
}

} // namespace ynn
