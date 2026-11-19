// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "ynnpack/base/arch.h"
#include "ynnpack/composites/composites_internal.h"

namespace ynn::detail {
namespace {

YNN_X86_TARGET_AVX2_FMA void rms_norm_avx2(size_t /*rows*/, size_t width, float epsilon, const float* input,
                                           const float* weight, const float* residual, bool has_scale, float scale,
                                           float* output, size_t row_start, size_t row_end) noexcept {
    YNN_DISABLE_FP_CONTRACT
    const float reciprocal = 1.0f / static_cast<float>(width);
    for (size_t row = row_start; row < row_end; ++row) {
        const float* x = input + row * width;
        float* y = output + row * width;
        size_t index = 0;
        __m256 p0 = _mm256_setzero_ps();
        for (; index + 8 <= width; index += 8) {
            const __m256 xv = _mm256_loadu_ps(x + index);
            p0 = _mm256_add_ps(p0, _mm256_mul_ps(xv, xv));
        }
        float partial[8];
        _mm256_storeu_ps(partial, p0);
        float sum = 0.0f;
        for (; index < width; ++index) sum += x[index] * x[index];
        for (size_t lane = 0; lane < 8; ++lane) sum += partial[lane];
        const float inverse = 1.0f / std::sqrt(sum * reciprocal + epsilon);
        const __m256 vinv = _mm256_set1_ps(inverse);
        if (residual == nullptr) {
            index = 0;
            for (; index + 8 <= width; index += 8) {
                _mm256_storeu_ps(y + index, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + index), vinv),
                                                          _mm256_loadu_ps(weight + index)));
            }
            for (; index < width; ++index) y[index] = x[index] * inverse * weight[index];
            continue;
        }
        const float* r = residual + row * width;
        const __m256 vscale = _mm256_set1_ps(scale);
        index = 0;
        for (; index + 8 <= width; index += 8) {
            __m256 v = _mm256_add_ps(
                _mm256_loadu_ps(r + index),
                _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + index), vinv), _mm256_loadu_ps(weight + index)));
            if (has_scale) v = _mm256_mul_ps(v, vscale);
            _mm256_storeu_ps(y + index, v);
        }
        for (; index < width; ++index) {
            const float value = r[index] + x[index] * inverse * weight[index];
            y[index] = has_scale ? value * scale : value;
        }
    }
}

} // namespace

bool arch_rms_norm_f32(size_t rows, size_t width, float epsilon, const float* input, const float* weight,
                       const float* residual, bool has_scale, float scale, float* output, size_t row_start,
                       size_t row_end) noexcept {
    if (!is_arch_supported(arch_flag::avx2_fma3)) return false;
    rms_norm_avx2(rows, width, epsilon, input, weight, residual, has_scale, scale, output, row_start, row_end);
    return true;
}

bool arch_rotary_f32(size_t, size_t, size_t, bool, const float*, const float*, const float*, float*, size_t,
                     size_t) noexcept {
    return false;
}

bool arch_rms_rotary_f32(size_t, size_t, size_t, bool, size_t, size_t, size_t, size_t, float, bool, const float*,
                         const float*, const float*, const float*, float*, size_t, size_t) noexcept {
    return false;
}

} // namespace ynn::detail
