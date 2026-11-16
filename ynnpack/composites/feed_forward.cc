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

#include "ynnpack/composites/composites.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/unary/unary.h"

namespace ynn {

void transpose_packed_subbyte_tile(int32_t bits, const uint8_t* source, size_t source_stride_bytes, size_t row_start,
                                   size_t row_count, size_t col_start, size_t col_count, uint8_t* destination) {
    const size_t values_per_byte = static_cast<size_t>(8 / bits);
    const size_t destination_row_bytes = (row_count + values_per_byte - 1) / values_per_byte;
    std::memset(destination, 0, col_count * destination_row_bytes);
    const uint8_t mask = static_cast<uint8_t>((1 << bits) - 1);
    for (size_t row = 0; row < row_count; ++row) {
        for (size_t column = 0; column < col_count; ++column) {
            const size_t source_column = col_start + column;
            const uint8_t value = static_cast<uint8_t>(
                (source[(row_start + row) * source_stride_bytes + source_column / values_per_byte] >>
                 ((source_column % values_per_byte) * bits)) &
                mask);
            destination[column * destination_row_bytes + row / values_per_byte] |=
                static_cast<uint8_t>(value << ((row % values_per_byte) * bits));
        }
    }
}

size_t gated_activation_tile_count(size_t rows, size_t width) {
    constexpr size_t kChunk = 1024;
    return rows * ((width + kChunk - 1) / kChunk);
}

ynn_status silu_gate_tiles_f32(size_t rows, size_t width, const float* input, float* output, size_t tile_start,
                               size_t tile_end) {
    static const auto sigmoid_kernel = get_unary_kernel(ynn_unary_sigmoid, ynn_type_fp32, ynn_type_fp32);
    static const auto sigmoid_params = get_unary_params(ynn_unary_sigmoid);
    if (!sigmoid_kernel) return ynn_status_unsupported_parameter;
    if (rows == 0 || width == 0) return ynn_status_success;
    constexpr size_t kChunk = 1024;
    const size_t tiles_per_row = (width + kChunk - 1) / kChunk;
    tile_end = std::min(tile_end, rows * tiles_per_row);
    for (size_t tile = tile_start; tile < tile_end; ++tile) {
        const size_t row = tile / tiles_per_row;
        const size_t offset = (tile % tiles_per_row) * kChunk;
        const size_t count = std::min(kChunk, width - offset);
        const float* gate = input + row * 2 * width + offset;
        const float* up = gate + width;
        float* hidden = output + row * width + offset;
        sigmoid_kernel(1, count, count * sizeof(float), gate, count * sizeof(float), hidden, &sigmoid_params);
        for (size_t i = 0; i < count; ++i) {
            hidden[i] = (hidden[i] * gate[i]) * up[i];
        }
    }
    return ynn_status_success;
}

ynn_status silu_gate_f32(size_t rows, size_t width, const float* input, float* output, size_t row_start,
                         size_t row_end) {
    row_end = std::min(row_end, rows);
    if (row_start >= row_end || width == 0) return ynn_status_success;
    constexpr size_t kChunk = 1024;
    const size_t tiles_per_row = (width + kChunk - 1) / kChunk;
    return silu_gate_tiles_f32(rows, width, input, output, row_start * tiles_per_row, row_end * tiles_per_row);
}

ynn_status gelu_gate_tiles_f32(size_t rows, size_t width, const float* input, float* output, size_t tile_start,
                               size_t tile_end) {
    static const auto tanh_kernel = get_unary_kernel(ynn_unary_tanh, ynn_type_fp32, ynn_type_fp32);
    static const auto tanh_params = get_unary_params(ynn_unary_tanh);
    if (!tanh_kernel) return ynn_status_unsupported_parameter;
    if (rows == 0 || width == 0) return ynn_status_success;
    constexpr float kScale = 0.79788456080286535588f; // sqrt(2 / pi)
    constexpr size_t kChunk = 1024;
    const size_t tiles_per_row = (width + kChunk - 1) / kChunk;
    tile_end = std::min(tile_end, rows * tiles_per_row);
    for (size_t tile = tile_start; tile < tile_end; ++tile) {
        const size_t row = tile / tiles_per_row;
        const size_t offset = (tile % tiles_per_row) * kChunk;
        const size_t count = std::min(kChunk, width - offset);
        const float* gate = input + row * 2 * width + offset;
        const float* up = gate + width;
        float* hidden = output + row * width + offset;
        {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
            for (size_t i = 0; i < count; ++i) {
                const float g = gate[i];
                hidden[i] = (g + (g * g * g) * 0.044715f) * kScale;
            }
        }
        tanh_kernel(1, count, count * sizeof(float), hidden, count * sizeof(float), hidden, &tanh_params);
        {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
            for (size_t i = 0; i < count; ++i) {
                hidden[i] = ((gate[i] * 0.5f) * (1.0f + hidden[i])) * up[i];
            }
        }
    }
    return ynn_status_success;
}

ynn_status gelu_gate_f32(size_t rows, size_t width, const float* input, float* output, size_t row_start,
                         size_t row_end) {
    row_end = std::min(row_end, rows);
    if (row_start >= row_end || width == 0) return ynn_status_success;
    constexpr size_t kChunk = 1024;
    const size_t tiles_per_row = (width + kChunk - 1) / kChunk;
    return gelu_gate_tiles_f32(rows, width, input, output, row_start * tiles_per_row, row_end * tiles_per_row);
}

size_t gated_multiply_tile_count(size_t elements) {
    constexpr size_t kChunk = 1024;
    return (elements + kChunk - 1) / kChunk;
}

ynn_status silu_multiply_tiles_f32(size_t elements, const float* gate, const float* up, float* output,
                                   size_t tile_start, size_t tile_end) {
    static const auto sigmoid_kernel = get_unary_kernel(ynn_unary_sigmoid, ynn_type_fp32, ynn_type_fp32);
    static const auto sigmoid_params = get_unary_params(ynn_unary_sigmoid);
    if (!sigmoid_kernel) return ynn_status_unsupported_parameter;
    if (elements == 0) return ynn_status_success;
    constexpr size_t kChunk = 1024;
    const size_t total_tiles = gated_multiply_tile_count(elements);
    tile_end = std::min(tile_end, total_tiles);
    alignas(64) float sig_buf[kChunk];
    for (size_t tile = tile_start; tile < tile_end; ++tile) {
        const size_t offset = tile * kChunk;
        const size_t count = std::min(kChunk, elements - offset);
        const float* g_ptr = gate + offset;
        const float* u_ptr = up + offset;
        float* dst = output + offset;
        sigmoid_kernel(1, count, count * sizeof(float), g_ptr, count * sizeof(float), sig_buf, &sigmoid_params);
        {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
            for (size_t i = 0; i < count; ++i) {
                dst[i] = (g_ptr[i] * sig_buf[i]) * u_ptr[i];
            }
        }
    }
    return ynn_status_success;
}

ynn_status gelu_multiply_tiles_f32(size_t elements, const float* gate, const float* up, float* output,
                                   size_t tile_start, size_t tile_end) {
    static const auto tanh_kernel = get_unary_kernel(ynn_unary_tanh, ynn_type_fp32, ynn_type_fp32);
    static const auto tanh_params = get_unary_params(ynn_unary_tanh);
    if (!tanh_kernel) return ynn_status_unsupported_parameter;
    if (elements == 0) return ynn_status_success;
    constexpr float kScale = 0.79788456080286535588f; // sqrt(2 / pi)
    constexpr size_t kChunk = 1024;
    const size_t total_tiles = gated_multiply_tile_count(elements);
    tile_end = std::min(tile_end, total_tiles);
    alignas(64) float inner_buf[kChunk];
    for (size_t tile = tile_start; tile < tile_end; ++tile) {
        const size_t offset = tile * kChunk;
        const size_t count = std::min(kChunk, elements - offset);
        const float* g_ptr = gate + offset;
        const float* u_ptr = up + offset;
        float* dst = output + offset;
        {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
            for (size_t i = 0; i < count; ++i) {
                const float g = g_ptr[i];
                inner_buf[i] = (g + (g * g * g) * 0.044715f) * kScale;
            }
        }
        tanh_kernel(1, count, count * sizeof(float), inner_buf, count * sizeof(float), inner_buf, &tanh_params);
        {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
            for (size_t i = 0; i < count; ++i) {
                dst[i] = ((g_ptr[i] * 0.5f) * (1.0f + inner_buf[i])) * u_ptr[i];
            }
        }
    }
    return ynn_status_success;
}

ynn_status packed_ffn_mid_tiles_f32(int32_t bits, size_t m, size_t input_size, size_t intermediate_size,
                                    float gate_output_scale, float down_input_scale, bool silu, const void* gate_packed,
                                    const void* up_packed, const int8_t* quantized_input, const float* input_scales,
                                    int8_t* quantized_mid, size_t tile_start, size_t tile_end) {
    const size_t mid_tiles = blockwise_linear_tile_count(intermediate_size);
    tile_end = std::min(tile_end, mid_tiles);
    if (tile_start >= tile_end || m == 0) return ynn_status_success;

    const size_t in_padded = symmetric_linear_padded_block(input_size);
    const size_t mid_padded = symmetric_linear_padded_block(intermediate_size);
    constexpr size_t M_STEP = 32;
    alignas(64) float gate_buf[M_STEP * 32];
    alignas(64) float up_buf[M_STEP * 32];

    for (size_t tile = tile_start; tile < tile_end; ++tile) {
        const size_t n0 = tile * 32;
        const size_t n_valid = std::min<size_t>(32, intermediate_size - n0);
        for (size_t m0 = 0; m0 < m; m0 += M_STEP) {
            const size_t m_count = std::min<size_t>(M_STEP, m - m0);
            if (n_valid < 32) {
                std::memset(gate_buf, 0, m_count * 32 * sizeof(float));
                std::memset(up_buf, 0, m_count * 32 * sizeof(float));
            }
            symmetric_linear_f32(bits, m_count, intermediate_size, input_size, input_size, gate_output_scale,
                                 gate_packed, quantized_input + m0 * in_padded, input_scales + m0, gate_buf, 32, tile,
                                 tile + 1);
            symmetric_linear_f32(bits, m_count, intermediate_size, input_size, input_size, gate_output_scale, up_packed,
                                 quantized_input + m0 * in_padded, input_scales + m0, up_buf, 32, tile, tile + 1);
            const size_t elems = m_count * 32;
            const ynn_status status = silu ? silu_multiply_tiles_f32(elems, gate_buf, up_buf, gate_buf, 0, 1)
                                           : gelu_multiply_tiles_f32(elems, gate_buf, up_buf, gate_buf, 0, 1);
            if (status != ynn_status_success) return status;
            if (n_valid < 32) {
                for (size_t r = 0; r < m_count; ++r)
                    std::memset(gate_buf + r * 32 + n_valid, 0, (32 - n_valid) * sizeof(float));
            }
            float dummy_scale = 0.0f;
            for (size_t r = 0; r < m_count; ++r) {
                quantize_symmetric_blocks_f32(bits, 1, 32, 32, down_input_scale, gate_buf + r * 32,
                                              quantized_mid + (m0 + r) * mid_padded + tile * 32, &dummy_scale, 0, 1);
            }
        }
    }
    return ynn_status_success;
}

void elementwise_binary_f32(bool is_multiply, size_t elements, size_t rhs_stride, const float* lhs, const float* rhs,
                            float* output, size_t start, size_t end) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    end = std::min(end, elements);
    if (start >= end) return;
    if (rhs_stride == 0) {
        const float scalar = rhs[0];
        if (is_multiply) {
            for (size_t i = start; i < end; ++i) output[i] = lhs[i] * scalar;
        } else {
            for (size_t i = start; i < end; ++i) output[i] = lhs[i] + scalar;
        }
        return;
    }
    if (rhs_stride == elements) {
        if (is_multiply) {
            for (size_t i = start; i < end; ++i) output[i] = lhs[i] * rhs[i];
        } else {
            for (size_t i = start; i < end; ++i) output[i] = lhs[i] + rhs[i];
        }
        return;
    }
    if (is_multiply) {
        for (size_t i = start; i < end; ++i) output[i] = lhs[i] * rhs[i % rhs_stride];
    } else {
        for (size_t i = start; i < end; ++i) output[i] = lhs[i] + rhs[i % rhs_stride];
    }
}

} // namespace ynn