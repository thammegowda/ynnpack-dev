// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"

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

}  // namespace ynn