// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>

#include "ynnpack/base/arch.h"
#include "ynnpack/composites/composites_internal.h"

namespace ynn::detail {
namespace {

YNN_NEON_TARGET_DOTPROD int32_t dot_i8_neon_dotprod(size_t size, const int8_t* query, const int8_t* key,
                                                    int32_t query_zero_point, int32_t key_zero_point) noexcept {
    int32_t query_sum = 0;
    int32_t key_sum = 0;
    int32x4_t dot = vdupq_n_s32(0);
    size_t channel = 0;
    for (; channel + 16 <= size; channel += 16) {
        const int8x16_t query_values = vld1q_s8(query + channel);
        const int8x16_t key_values = vld1q_s8(key + channel);
        dot = vdotq_s32(dot, query_values, key_values);
        if (key_zero_point != 0) {
            query_sum += vaddlvq_s8(query_values);
        }
        key_sum += vaddlvq_s8(key_values);
    }
    int32_t result = vaddvq_s32(dot);
    if (channel + 8 <= size) {
        const int8x8_t query_values = vld1_s8(query + channel);
        const int8x8_t key_values = vld1_s8(key + channel);
        result += vaddlvq_s16(vmull_s8(query_values, key_values));
        if (key_zero_point != 0) {
            query_sum += vaddlv_s8(query_values);
        }
        key_sum += vaddlv_s8(key_values);
        channel += 8;
    }
    for (; channel < size; ++channel) {
        result += query[channel] * key[channel];
        if (key_zero_point != 0) {
            query_sum += query[channel];
        }
        key_sum += key[channel];
    }
    return result - query_zero_point * key_sum - key_zero_point * query_sum +
           static_cast<int32_t>(size) * query_zero_point * key_zero_point;
}

} // namespace

bool arch_dot_i8_with_zp(size_t size, const int8_t* query, const int8_t* key, int32_t query_zero_point,
                         int32_t key_zero_point, int32_t* out_dot) noexcept {
    if (!is_arch_supported(arch_flag::neondot)) return false;
    *out_dot = dot_i8_neon_dotprod(size, query, key, query_zero_point, key_zero_point);
    return true;
}

size_t arch_scale_f32(size_t size, float scale, float* output) noexcept {
    const float32x4_t scale_vector = vdupq_n_f32(scale);
    size_t channel = 0;
    for (; channel + 8 <= size; channel += 8) {
        vst1q_f32(output + channel, vmulq_f32(vld1q_f32(output + channel), scale_vector));
        vst1q_f32(output + channel + 4, vmulq_f32(vld1q_f32(output + channel + 4), scale_vector));
    }
    return channel;
}

size_t arch_accumulate_i8_value_f32(size_t size, const int8_t* value, int32_t zero_point, float scale, float weight,
                                    float* output) noexcept {
    const int32x4_t zero = vdupq_n_s32(zero_point);
    const float32x4_t multiplier = vdupq_n_f32(scale * weight);
    size_t channel = 0;
    for (; channel + 8 <= size; channel += 8) {
        const int16x8_t value_i16 = vmovl_s8(vld1_s8(value + channel));
        const float32x4_t value0 = vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_low_s16(value_i16)), zero));
        const float32x4_t value1 = vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_high_s16(value_i16)), zero));
        vst1q_f32(output + channel, vfmaq_f32(vld1q_f32(output + channel), value0, multiplier));
        vst1q_f32(output + channel + 4, vfmaq_f32(vld1q_f32(output + channel + 4), value1, multiplier));
    }
    return channel;
}

bool arch_dot_f32_lanes8(const float* left, const float* right, size_t count, float* out_sum) noexcept {
    YNN_DISABLE_FP_CONTRACT
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    size_t index = 0;
    for (; index + 8 <= count; index += 8) {
        acc0 = vaddq_f32(acc0, vmulq_f32(vld1q_f32(left + index), vld1q_f32(right + index)));
        acc1 = vaddq_f32(acc1, vmulq_f32(vld1q_f32(left + index + 4), vld1q_f32(right + index + 4)));
    }
    float lanes[8];
    vst1q_f32(lanes, acc0);
    vst1q_f32(lanes + 4, acc1);
    float sum = 0.0f;
    for (size_t lane = 0; lane < 8; ++lane) sum += lanes[lane];
    for (; index < count; ++index) sum += left[index] * right[index];
    *out_sum = sum;
    return true;
}

size_t arch_accumulate_weighted_f32(float* dst, const float* src, float weight, size_t count) noexcept {
    YNN_DISABLE_FP_CONTRACT
    const float32x4_t vw = vdupq_n_f32(weight);
    size_t index = 0;
    for (; index + 8 <= count; index += 8) {
        vst1q_f32(dst + index, vaddq_f32(vld1q_f32(dst + index), vmulq_f32(vld1q_f32(src + index), vw)));
        vst1q_f32(dst + index + 4, vaddq_f32(vld1q_f32(dst + index + 4), vmulq_f32(vld1q_f32(src + index + 4), vw)));
    }
    return index;
}

} // namespace ynn::detail
