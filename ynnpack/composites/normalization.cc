// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/composites_internal.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

void rms_norm_f32(size_t rows, size_t width, float epsilon, const float* input, const float* weight,
                  const float* residual, bool has_scale, float scale, float* output, size_t row_start, size_t row_end) {
    YNN_DISABLE_FP_CONTRACT
    row_end = std::min(row_end, rows);
    if (detail::arch_rms_norm_f32(rows, width, epsilon, input, weight, residual, has_scale, scale, output, row_start,
                                  row_end)) {
        return;
    }
    const float reciprocal = 1.0f / static_cast<float>(width);
    for (size_t row = row_start; row < row_end; ++row) {
        const float* x = input + row * width;
        float* y = output + row * width;
        size_t index = 0;
        float partial[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (; index + 8 <= width; index += 8) {
            for (size_t lane = 0; lane < 8; ++lane) partial[lane] += x[index + lane] * x[index + lane];
        }
        float sum = 0.0f;
        for (; index < width; ++index) sum += x[index] * x[index];
        for (size_t lane = 0; lane < 8; ++lane) sum += partial[lane];
        const float inverse = 1.0f / std::sqrt(sum * reciprocal + epsilon);
        if (residual == nullptr) {
            for (size_t col = 0; col < width; ++col) y[col] = x[col] * inverse * weight[col];
            continue;
        }
        const float* r = residual + row * width;
        for (size_t col = 0; col < width; ++col) {
            const float value = r[col] + x[col] * inverse * weight[col];
            y[col] = has_scale ? value * scale : value;
        }
    }
}

void rotary_f32(size_t sequence_rows, size_t heads, size_t width, bool shared_angles, const float* input,
                const float* cosine, const float* sine, float* output, size_t row_start, size_t row_end) {
    YNN_DISABLE_FP_CONTRACT
    if (detail::arch_rotary_f32(sequence_rows, heads, width, shared_angles, input, cosine, sine, output, row_start,
                                row_end)) {
        return;
    }
    const size_t half = width / 2;
    for (size_t row = row_start; row < row_end; ++row) {
        const size_t angle = (shared_angles ? row % sequence_rows : row) * half;
        const float* c = cosine + angle;
        const float* s = sine + angle;
        for (size_t head = 0; head < heads; ++head) {
            const float* x = input + (row * heads + head) * width;
            float* y = output + (row * heads + head) * width;
            for (size_t index = 0; index < half; ++index) {
                const float first = x[index];
                const float second = x[index + half];
                y[index] = first * c[index] - second * s[index];
                y[index + half] = second * c[index] + first * s[index];
            }
        }
    }
}

void rms_rotary_f32(size_t sequence_rows, size_t heads, size_t width, bool sliced, size_t input_rows,
                    size_t input_stride, size_t sequence_offset, size_t channel_offset, float epsilon,
                    bool shared_angles, const float* input, const float* weight, const float* cosine, const float* sine,
                    float* output, size_t vec_start, size_t vec_end) {
    YNN_DISABLE_FP_CONTRACT
    if (detail::arch_rms_rotary_f32(sequence_rows, heads, width, sliced, input_rows, input_stride, sequence_offset,
                                    channel_offset, epsilon, shared_angles, input, weight, cosine, sine, output,
                                    vec_start, vec_end)) {
        return;
    }
    const size_t half = width / 2;
    const float reciprocal = 1.0f / static_cast<float>(width);
    for (size_t vector = vec_start; vector < vec_end; ++vector) {
        const size_t sample = vector / (sequence_rows * heads);
        const size_t position = (vector / heads) % sequence_rows;
        const size_t head = vector % heads;
        const float* x = sliced ? input + (sample * input_rows + sequence_offset + position) * input_stride +
                                      channel_offset + head * width
                                : input + vector * width;
        float* y = output + vector * width;
        size_t index = 0;
        float partial[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (; index + 8 <= width; index += 8) {
            for (size_t lane = 0; lane < 8; ++lane) partial[lane] += x[index + lane] * x[index + lane];
        }
        float sum = 0.0f;
        for (; index < width; ++index) sum += x[index] * x[index];
        for (size_t lane = 0; lane < 8; ++lane) sum += partial[lane];
        const float inverse = 1.0f / std::sqrt(sum * reciprocal + epsilon);
        const size_t angle = (shared_angles ? position : sample * sequence_rows + position) * half;
        const float* c = cosine + angle;
        const float* s = sine + angle;
        for (size_t col = 0; col < half; ++col) {
            const float first = x[col] * inverse * weight[col];
            const float second = x[col + half] * inverse * weight[col + half];
            y[col] = first * c[col] - second * s[col];
            y[col + half] = second * c[col] + first * s[col];
        }
    }
}

} // namespace ynn
