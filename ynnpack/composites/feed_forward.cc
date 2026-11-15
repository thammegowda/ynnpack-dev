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
#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/unary/unary.h"

namespace ynn {
namespace {

ynn_status define_calibration(ynn_subgraph_t subgraph, uint32_t input_id,
                              uint32_t scale_id, uint32_t& output_id) {
  uint32_t scaled_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_divide, input_id,
                                        scale_id, &scaled_id, 0));
  uint32_t rounded_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_unary(subgraph, ynn_unary_round, scaled_id,
                                       &rounded_id, 0));
  uint32_t low_id = YNN_INVALID_VALUE_ID;
  uint32_t high_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(define_constant(subgraph, -128.0f, low_id));
  YNN_RETURN_IF_ERROR(define_constant(subgraph, 127.0f, high_id));
  uint32_t lower_id = YNN_INVALID_VALUE_ID;
  uint32_t clamped_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_max, rounded_id,
                                        low_id, &lower_id, 0));
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_min, lower_id,
                                        high_id, &clamped_id, 0));
  return ynn_define_binary(subgraph, ynn_binary_multiply, clamped_id, scale_id,
                           &output_id, 0);
}

ynn_status define_quantize(ynn_subgraph_t subgraph, uint32_t input_id,
                           uint32_t scale_id, uint32_t zero_id,
                           uint32_t& output_id) {
  return ynn_define_quantize(subgraph, input_id, ynn_type_int8, zero_id,
                             scale_id, &output_id, 0);
}

}  // namespace

ynn_status define_packed_feed_forward(
    ynn_subgraph_t subgraph, uint32_t input_id, uint32_t input_scale_id,
    const uint32_t* gate_weight_ids, const uint32_t* gate_scale_ids,
    const uint32_t* up_weight_ids, const uint32_t* up_scale_ids,
    uint32_t first_output_scale_id, const uint32_t* down_weight_ids,
    uint32_t down_scale_id, uint32_t down_input_scale_id,
    uint32_t down_output_scale_id, size_t input_size, size_t intermediate_size,
    size_t tile_size, bool gated, uint32_t& output_id) {
  if (input_size == 0 || intermediate_size == 0 || tile_size == 0 ||
      gate_weight_ids == nullptr || gate_scale_ids == nullptr ||
      down_weight_ids == nullptr ||
      (gated && (up_weight_ids == nullptr || up_scale_ids == nullptr))) {
    return ynn_status_invalid_parameter;
  }
  const size_t num_tiles = (intermediate_size + tile_size - 1) / tile_size;
  int32_t zero = 0;
  uint32_t zero_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_tensor(subgraph, ynn_type_int32, 0, nullptr,
                                        &zero, YNN_VALUE_FLAG_COPY_DATA,
                                        &zero_id));
  uint32_t quantized_input_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(define_quantize(subgraph, input_id, input_scale_id,
                                      zero_id, quantized_input_id));
  uint32_t accumulated_id = YNN_INVALID_VALUE_ID;
  for (size_t tile = 0, begin = 0; tile < num_tiles;
       ++tile, begin += tile_size) {
    const size_t length = std::min(tile_size, intermediate_size - begin);
    uint32_t gate_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(define_blockwise_dot(
        subgraph, quantized_input_id, zero_id, input_scale_id,
        gate_weight_ids[tile], YNN_INVALID_VALUE_ID, gate_scale_ids[tile],
        input_size, YNN_INVALID_VALUE_ID, ynn_type_fp32, gate_id));
    uint32_t calibrated_gate_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(define_calibration(subgraph, gate_id,
                         first_output_scale_id,
                         calibrated_gate_id));

    uint32_t activated_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(
      define_approx_gelu(subgraph, calibrated_gate_id, activated_id));
    if (gated) {
      uint32_t up_id = YNN_INVALID_VALUE_ID;
      YNN_RETURN_IF_ERROR(define_blockwise_dot(
          subgraph, quantized_input_id, zero_id, input_scale_id,
        up_weight_ids[tile], YNN_INVALID_VALUE_ID, up_scale_ids[tile],
        input_size, YNN_INVALID_VALUE_ID, ynn_type_fp32, up_id));
      uint32_t calibrated_up_id = YNN_INVALID_VALUE_ID;
      YNN_RETURN_IF_ERROR(define_calibration(subgraph, up_id,
                                             first_output_scale_id,
                                             calibrated_up_id));
      uint32_t product_id = YNN_INVALID_VALUE_ID;
      YNN_RETURN_IF_ERROR(ynn_define_binary(
          subgraph, ynn_binary_multiply, activated_id, calibrated_up_id,
          &product_id, 0));
      activated_id = product_id;
    }

    uint32_t quantized_activation_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(define_quantize(subgraph, activated_id,
                      down_input_scale_id, zero_id,
                      quantized_activation_id));
    uint32_t partial_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_dot(
        subgraph, 1, quantized_activation_id, down_weight_ids[tile],
        accumulated_id, &partial_id, 0));
    accumulated_id = partial_id;
  }
  const int32_t transpose[] = {1, 0};
  uint32_t aligned_down_scale_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_static_transpose(
      subgraph, 2, transpose, down_scale_id, &aligned_down_scale_id, 0));
  uint32_t weight_scaled_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_binary(
      subgraph, ynn_binary_multiply, accumulated_id, aligned_down_scale_id,
      &weight_scaled_id, 0));
  uint32_t fully_scaled_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_binary(
      subgraph, ynn_binary_multiply, weight_scaled_id, down_input_scale_id,
      &fully_scaled_id, 0));
  return define_calibration(subgraph, fully_scaled_id, down_output_scale_id,
                            output_id);
}

void transpose_packed_subbyte_tile(int32_t bits, const uint8_t* source, size_t source_stride_bytes, size_t row_start,
                                   size_t row_count, size_t col_start, size_t col_count, uint8_t* destination) {
  const size_t values_per_byte = static_cast<size_t>(8 / bits);
  const size_t destination_row_bytes = row_count / values_per_byte;
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

}  // namespace ynn