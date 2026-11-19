// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "ynnpack/composites/composites_internal.h"

namespace ynn::detail {

bool arch_accumulate_grouped_value(size_t size, const int8_t* value, int32_t zero_point, float scale,
                                   const float* probabilities, size_t probability_stride, size_t heads, float* output,
                                   size_t output_stride) noexcept {
    const int32x4_t zero = vdupq_n_s32(zero_point);
    const float32x4_t scale_vector = vdupq_n_f32(scale);
    size_t channel = 0;
    for (; channel + 8 <= size; channel += 8) {
        const int16x8_t value_i16 = vmovl_s8(vld1_s8(value + channel));
        const float32x4_t value0 =
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_low_s16(value_i16)), zero)), scale_vector);
        const float32x4_t value1 =
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_high_s16(value_i16)), zero)), scale_vector);
        for (size_t head = 0; head < heads; ++head) {
            const float32x4_t weight = vdupq_n_f32(probabilities[head * probability_stride]);
            float* head_output = output + head * output_stride + channel;
            vst1q_f32(head_output, vfmaq_f32(vld1q_f32(head_output), value0, weight));
            vst1q_f32(head_output + 4, vfmaq_f32(vld1q_f32(head_output + 4), value1, weight));
        }
    }
    for (; channel < size; ++channel) {
        const float dequantized = static_cast<float>(value[channel] - zero_point) * scale;
        for (size_t head = 0; head < heads; ++head) {
            output[head * output_stride + channel] += probabilities[head * probability_stride] * dequantized;
        }
    }
    return true;
}

bool arch_score_grouped_key_token(size_t head_dim, size_t heads_per_group, const float* query_tile,
                                  const int8_t* key_token, const blockwise_quantization_f32_params& key_quantization,
                                  float* scores) noexcept {
    constexpr size_t MAX_HEADS_PER_GROUP = 8;
    float32x4_t vector_scores[MAX_HEADS_PER_GROUP / 4]{};
    const size_t blocks = (head_dim + key_quantization.block_size - 1) / key_quantization.block_size;
    for (size_t block = 0; block < blocks; ++block) {
        const size_t channel_start = block * key_quantization.block_size;
        const size_t channel_end = std::min(head_dim, channel_start + key_quantization.block_size);
        const float key_scale = key_quantization.scales[block];
        const int32_t key_zero_point = key_quantization.zero_points[block];
        for (size_t channel = channel_start; channel < channel_end; ++channel) {
            const float dequantized = static_cast<float>(key_token[channel] - key_zero_point) * key_scale;
            size_t head = 0;
            for (; head + 4 <= heads_per_group; head += 4) {
                vector_scores[head / 4] = vfmaq_n_f32(
                    vector_scores[head / 4], vld1q_f32(query_tile + channel * heads_per_group + head), dequantized);
            }
            for (; head < heads_per_group; ++head) {
                scores[head] = std::fma(query_tile[channel * heads_per_group + head], dequantized, scores[head]);
            }
        }
    }
    for (size_t head = 0; head + 4 <= heads_per_group; head += 4) {
        vst1q_f32(scores + head, vector_scores[head / 4]);
    }
    return true;
}

} // namespace ynn::detail
