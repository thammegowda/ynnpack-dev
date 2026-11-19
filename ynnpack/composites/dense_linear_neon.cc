// Licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "ynnpack/base/arch.h"
#include "ynnpack/composites/composites_internal.h"

namespace ynn::detail {
namespace {

YNN_NEON_TARGET_BF16 size_t f32_to_bf16_neon_bf16(const float* src, uint16_t* dst, size_t count) noexcept {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const float32x4_t v0 = vld1q_f32(src + i);
        const float32x4_t v1 = vld1q_f32(src + i + 4);
        vst1q_u16(dst + i, vreinterpretq_u16_bf16(vcombine_bf16(vcvt_bf16_f32(v0), vcvt_bf16_f32(v1))));
    }
    return i;
}

} // namespace

size_t arch_f32_to_bf16_slice(const float* src, uint16_t* dst, size_t count) noexcept {
    if (is_arch_supported(arch_flag::neonbf16)) {
        return f32_to_bf16_neon_bf16(src, dst, count);
    }
    return 0;
}

size_t arch_bf16_to_f32_slice(const uint16_t* src, float* dst, size_t count) noexcept {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const uint16x8_t v = vld1q_u16(src + i);
        const uint32x4_t lo = vshlq_n_u32(vmovl_u16(vget_low_u16(v)), 16);
        const uint32x4_t hi = vshlq_n_u32(vmovl_u16(vget_high_u16(v)), 16);
        vst1q_f32(dst + i, vreinterpretq_f32_u32(lo));
        vst1q_f32(dst + i + 4, vreinterpretq_f32_u32(hi));
    }
    return i;
}

bool arch_quantize_dynamic_int8_rows_f32(size_t /*rows*/, size_t k, const float* input, int8_t* quantized,
                                         float* scales, int32_t* zero_points, size_t row_start,
                                         size_t row_end) noexcept {
    for (size_t row = row_start; row < row_end; ++row) {
        const float* x = input + row * k;
        float32x4_t vmin0 = vdupq_n_f32(0.0f), vmin1 = vdupq_n_f32(0.0f);
        float32x4_t vmax0 = vdupq_n_f32(0.0f), vmax1 = vdupq_n_f32(0.0f);
        size_t col = 0;
        for (; col + 8 <= k; col += 8) {
            const float32x4_t x0 = vld1q_f32(x + col);
            const float32x4_t x1 = vld1q_f32(x + col + 4);
            vmin0 = vminq_f32(vmin0, x0);
            vmin1 = vminq_f32(vmin1, x1);
            vmax0 = vmaxq_f32(vmax0, x0);
            vmax1 = vmaxq_f32(vmax1, x1);
        }
        float rmin = vminvq_f32(vminq_f32(vmin0, vmin1));
        float rmax = vmaxvq_f32(vmaxq_f32(vmax0, vmax1));
        for (; col < k; ++col) {
            rmin = std::min(rmin, x[col]);
            rmax = std::max(rmax, x[col]);
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
        scales[row] = scale;
        zero_points[row] = zp;
        int8_t* q = quantized + row * k;
        col = 0;
        const float32x4_t vinv = vdupq_n_f32(inv_scale);
        const int32x4_t vzp = vdupq_n_s32(zp);
        for (; col + 16 <= k; col += 16) {
            const int32x4_t i0 = vaddq_s32(vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x + col + 0), vinv)), vzp);
            const int32x4_t i1 = vaddq_s32(vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x + col + 4), vinv)), vzp);
            const int32x4_t i2 = vaddq_s32(vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x + col + 8), vinv)), vzp);
            const int32x4_t i3 = vaddq_s32(vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x + col + 12), vinv)), vzp);
            const int16x8_t s0 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
            const int16x8_t s1 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
            vst1q_s8(q + col, vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1)));
        }
        for (; col < k; ++col) {
            const int32_t v = static_cast<int32_t>(std::lround(x[col] * inv_scale)) + zp;
            q[col] = static_cast<int8_t>(std::clamp<int32_t>(v, -128, 127));
        }
    }
    return true;
}

size_t arch_dequantize_i32_row_f32(size_t n_valid, const int32_t* acc_row, const int32_t* wsum, float a_scale,
                                   int32_t a_zp, const float* ws, const float* bp, float* out_row) noexcept {
    const float32x4_t vas = vdupq_n_f32(a_scale);
    const int32x4_t vazp = vdupq_n_s32(a_zp);
    size_t col = 0;
    for (; col + 4 <= n_valid; col += 4) {
        const int32x4_t corr = vmlsq_s32(vld1q_s32(acc_row + col), vld1q_s32(wsum + col), vazp);
        const float32x4_t prod = vmulq_f32(vcvtq_f32_s32(corr), vmulq_f32(vld1q_f32(ws + col), vas));
        const float32x4_t res = bp ? vaddq_f32(prod, vld1q_f32(bp + col)) : prod;
        vst1q_f32(out_row + col, res);
    }
    return col;
}

} // namespace ynn::detail
