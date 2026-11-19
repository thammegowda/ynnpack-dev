// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>

#include "ynnpack/composites/composites_internal.h"

namespace ynn::detail {

size_t arch_quantize_symmetric_int8_f32(size_t count, float scale, const float* input, int8_t* output) noexcept {
    const float32x4_t vscale = vdupq_n_f32(scale);
    size_t index = 0;
    for (; index + 16 <= count; index += 16) {
        const int32x4_t i0 = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(input + index + 0), vscale));
        const int32x4_t i1 = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(input + index + 4), vscale));
        const int32x4_t i2 = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(input + index + 8), vscale));
        const int32x4_t i3 = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(input + index + 12), vscale));
        const int16x8_t s0 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
        const int16x8_t s1 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
        vst1q_s8(output + index, vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1)));
    }
    return index;
}

size_t arch_argmax_chunk_f32(size_t count, const float* row_logits, size_t col_offset, float& best, int32_t& token,
                             bool& invalid) noexcept {
    uint32x4_t nan_or_pinf = vdupq_n_u32(0);
    const uint32x4_t exp_mask = vdupq_n_u32(0x7F800000u);
    const uint32x4_t ninf_bits = vdupq_n_u32(0xFF800000u);
    size_t column = 0;
    for (; column + 16 <= count; column += 16) {
        const float32x4_t v0 = vld1q_f32(row_logits + column + 0);
        const float32x4_t v1 = vld1q_f32(row_logits + column + 4);
        const float32x4_t v2 = vld1q_f32(row_logits + column + 8);
        const float32x4_t v3 = vld1q_f32(row_logits + column + 12);
        const uint32x4_t u0 = vreinterpretq_u32_f32(v0);
        const uint32x4_t u1 = vreinterpretq_u32_f32(v1);
        const uint32x4_t u2 = vreinterpretq_u32_f32(v2);
        const uint32x4_t u3 = vreinterpretq_u32_f32(v3);
        const uint32x4_t bad0 =
            vandq_u32(vceqq_u32(vandq_u32(u0, exp_mask), exp_mask), vmvnq_u32(vceqq_u32(u0, ninf_bits)));
        const uint32x4_t bad1 =
            vandq_u32(vceqq_u32(vandq_u32(u1, exp_mask), exp_mask), vmvnq_u32(vceqq_u32(u1, ninf_bits)));
        const uint32x4_t bad2 =
            vandq_u32(vceqq_u32(vandq_u32(u2, exp_mask), exp_mask), vmvnq_u32(vceqq_u32(u2, ninf_bits)));
        const uint32x4_t bad3 =
            vandq_u32(vceqq_u32(vandq_u32(u3, exp_mask), exp_mask), vmvnq_u32(vceqq_u32(u3, ninf_bits)));
        nan_or_pinf = vorrq_u32(nan_or_pinf, vorrq_u32(vorrq_u32(bad0, bad1), vorrq_u32(bad2, bad3)));
        const float block_max = vmaxvq_f32(vmaxq_f32(vmaxq_f32(v0, v1), vmaxq_f32(v2, v3)));
        if (block_max > best) {
            for (size_t i = 0; i < 16; ++i) {
                const float value = row_logits[column + i];
                if (value > best) {
                    best = value;
                    token = static_cast<int32_t>(col_offset + column + i);
                }
            }
        }
    }
    if (vmaxvq_u32(nan_or_pinf) != 0) invalid = true;
    return column;
}

} // namespace ynn::detail
