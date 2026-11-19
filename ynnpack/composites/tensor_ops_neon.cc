// Licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "ynnpack/composites/composites_internal.h"

namespace ynn::detail {

bool arch_layer_norm_f32(size_t /*rows*/, size_t width, float epsilon, const float* input, const float* residual_in,
                         const float* weight, const float* bias, float* output_norm, float* output_residual,
                         size_t row_start, size_t row_end) noexcept {
    const float inv_width = 1.0f / static_cast<float>(width);
    for (size_t r = row_start; r < row_end; ++r) {
        const float* x = input + r * width;
        const float* res = residual_in != nullptr ? residual_in + r * width : nullptr;
        float* y = output_norm + r * width;
        float* y_res = output_residual != nullptr ? output_residual + r * width : nullptr;

        float32x4_t vsum = vdupq_n_f32(0.0f);
        float sum = 0.0f;
        size_t i = 0;
        if (res != nullptr) {
            for (; i + 4 <= width; i += 4) {
                const float32x4_t v = vaddq_f32(vld1q_f32(x + i), vld1q_f32(res + i));
                if (y_res != nullptr) vst1q_f32(y_res + i, v);
                vsum = vaddq_f32(vsum, v);
            }
            sum = vaddvq_f32(vsum);
            for (; i < width; ++i) {
                const float v = x[i] + res[i];
                if (y_res != nullptr) y_res[i] = v;
                sum += v;
            }
        } else {
            for (; i + 4 <= width; i += 4) vsum = vaddq_f32(vsum, vld1q_f32(x + i));
            sum = vaddvq_f32(vsum);
            for (; i < width; ++i) sum += x[i];
        }
        const float mean = sum * inv_width;

        const float* src = y_res != nullptr ? y_res : x;
        float32x4_t vvsum = vdupq_n_f32(0.0f);
        const float32x4_t vmean = vdupq_n_f32(mean);
        size_t j = 0;
        for (; j + 4 <= width; j += 4) {
            const float32x4_t v = res != nullptr && y_res == nullptr ? vaddq_f32(vld1q_f32(x + j), vld1q_f32(res + j))
                                                                     : vld1q_f32(src + j);
            const float32x4_t d = vsubq_f32(v, vmean);
            vvsum = vfmaq_f32(vvsum, d, d);
        }
        float var_sum = vaddvq_f32(vvsum);
        for (; j < width; ++j) {
            const float v = (res != nullptr && y_res == nullptr) ? (x[j] + res[j]) : src[j];
            const float d = v - mean;
            var_sum += d * d;
        }
        const float inv_std = 1.0f / std::sqrt(var_sum * inv_width + epsilon);

        const float32x4_t vinv = vdupq_n_f32(inv_std);
        size_t k = 0;
        for (; k + 4 <= width; k += 4) {
            const float32x4_t v = res != nullptr && y_res == nullptr ? vaddq_f32(vld1q_f32(x + k), vld1q_f32(res + k))
                                                                     : vld1q_f32(src + k);
            const float32x4_t norm = vmulq_f32(vsubq_f32(v, vmean), vinv);
            const float32x4_t scaled = vfmaq_f32(vld1q_f32(bias + k), norm, vld1q_f32(weight + k));
            vst1q_f32(y + k, scaled);
        }
        for (; k < width; ++k) {
            const float v = (res != nullptr && y_res == nullptr) ? (x[k] + res[k]) : src[k];
            y[k] = (v - mean) * inv_std * weight[k] + bias[k];
        }
    }
    return true;
}

bool arch_softmax_f32(size_t /*rows*/, size_t width, bool logarithmic, const float* input, float* output,
                      size_t row_start, size_t row_end) noexcept {
    for (size_t r = row_start; r < row_end; ++r) {
        const float* x = input + r * width;
        float* y = output + r * width;

        float32x4_t vmax = vdupq_n_f32(-std::numeric_limits<float>::infinity());
        size_t i = 0;
        for (; i + 4 <= width; i += 4) vmax = vmaxq_f32(vmax, vld1q_f32(x + i));
        float max_val = vmaxvq_f32(vmax);
        for (; i < width; ++i) max_val = std::max(max_val, x[i]);

        float sum_exp = 0.0f;
        for (size_t j = 0; j < width; ++j) {
            const float e = std::exp(x[j] - max_val);
            if (!logarithmic) y[j] = e;
            sum_exp += e;
        }
        if (logarithmic) {
            const float log_z = max_val + std::log(sum_exp);
            for (size_t j = 0; j < width; ++j) y[j] = x[j] - log_z;
        } else {
            const float inv_sum = 1.0f / sum_exp;
            const float32x4_t vinv = vdupq_n_f32(inv_sum);
            size_t j = 0;
            for (; j + 4 <= width; j += 4) vst1q_f32(y + j, vmulq_f32(vld1q_f32(y + j), vinv));
            for (; j < width; ++j) y[j] *= inv_sum;
        }
    }
    return true;
}

size_t arch_broadcast_binary_inner_f32(bool is_multiply, size_t chunk, size_t sa, size_t sb, const float* ap,
                                       const float* bp, float* op) noexcept {
    size_t i = 0;
    if (sa == 1 && sb == 1) {
        if (is_multiply) {
            for (; i + 4 <= chunk; i += 4) vst1q_f32(op + i, vmulq_f32(vld1q_f32(ap + i), vld1q_f32(bp + i)));
        } else {
            for (; i + 4 <= chunk; i += 4) vst1q_f32(op + i, vaddq_f32(vld1q_f32(ap + i), vld1q_f32(bp + i)));
        }
    } else if (sa == 1 && sb == 0) {
        const float32x4_t vb = vdupq_n_f32(*bp);
        if (is_multiply) {
            for (; i + 4 <= chunk; i += 4) vst1q_f32(op + i, vmulq_f32(vld1q_f32(ap + i), vb));
        } else {
            for (; i + 4 <= chunk; i += 4) vst1q_f32(op + i, vaddq_f32(vld1q_f32(ap + i), vb));
        }
    } else if (sa == 0 && sb == 1) {
        const float32x4_t va = vdupq_n_f32(*ap);
        if (is_multiply) {
            for (; i + 4 <= chunk; i += 4) vst1q_f32(op + i, vmulq_f32(va, vld1q_f32(bp + i)));
        } else {
            for (; i + 4 <= chunk; i += 4) vst1q_f32(op + i, vaddq_f32(va, vld1q_f32(bp + i)));
        }
    }
    return i;
}

} // namespace ynn::detail
