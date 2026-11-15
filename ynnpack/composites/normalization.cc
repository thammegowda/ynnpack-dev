// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

ynn_status define_softmax(ynn_subgraph_t subgraph, uint32_t input_id,
                          float beta, uint32_t& output_id) {
  uint32_t scaled_input_id = input_id;
  if (beta != 1.0f) {
    // Multiply input by beta.
    uint32_t beta_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(define_constant(subgraph, beta, beta_id));
    uint32_t multiply_output_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply,
                                          input_id, beta_id,
                                          &multiply_output_id, 0));
    scaled_input_id = multiply_output_id;
  }

  // 1. Max reduction along last axis.
  uint32_t max_input_id = YNN_INVALID_VALUE_ID;
  const int32_t last_axis[] = {-1};
  YNN_RETURN_IF_ERROR(ynn_define_reduce(
      subgraph, ynn_reduce_max, 1, last_axis, scaled_input_id,
      YNN_INVALID_VALUE_ID, &max_input_id, YNN_NODE_FLAG_KEEP_DIMS));

  // 2. Subtract max from scaled input.
  uint32_t input_minus_max_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_subtract,
                                        scaled_input_id, max_input_id,
                                        &input_minus_max_id, 0));

  // 3. Exp.
  uint32_t exp_input_minus_max_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_unary(
      subgraph, ynn_unary_exp, input_minus_max_id, &exp_input_minus_max_id, 0));

  // 4. Sum reduction of exp along last axis.
  uint32_t sum_exp_input_minus_max_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(
      ynn_define_reduce(subgraph, ynn_reduce_sum, 1, last_axis,
                        exp_input_minus_max_id, YNN_INVALID_VALUE_ID,
                        &sum_exp_input_minus_max_id, YNN_NODE_FLAG_KEEP_DIMS));

  // 5. Reciprocal of sum (1.0 / sum).
  uint32_t one_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(define_constant(subgraph, 1.0f, one_id));

  uint32_t inv_sum_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_divide, one_id,
                                        sum_exp_input_minus_max_id, &inv_sum_id,
                                        0));

  // 6. Multiply exp by reciprocal of sum.
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply,
                                        exp_input_minus_max_id, inv_sum_id,
                                        &output_id, 0));

  return ynn_status_success;
}

ynn_status define_log_softmax(ynn_subgraph_t subgraph, uint32_t input_id,
                              uint32_t& output_id) {
  // LogSoftmax(x)_i = x_i - max(x) - log(sum(exp(x_j - max(x))))

  // 1. Max reduction along last axis.
  uint32_t max_input_id = YNN_INVALID_VALUE_ID;
  const int32_t last_axis[] = {-1};
  YNN_RETURN_IF_ERROR(ynn_define_reduce(
      subgraph, ynn_reduce_max, 1, last_axis, input_id, YNN_INVALID_VALUE_ID,
      &max_input_id, YNN_NODE_FLAG_KEEP_DIMS));

  // 2. Subtract max from input.
  uint32_t input_minus_max_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_subtract, input_id,
                                        max_input_id, &input_minus_max_id, 0));

  // 3. Exp.
  uint32_t exp_input_minus_max_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_unary(
      subgraph, ynn_unary_exp, input_minus_max_id, &exp_input_minus_max_id, 0));

  // 4. Sum reduction of exp along last axis.
  uint32_t sum_exp_input_minus_max_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(
      ynn_define_reduce(subgraph, ynn_reduce_sum, 1, last_axis,
                        exp_input_minus_max_id, YNN_INVALID_VALUE_ID,
                        &sum_exp_input_minus_max_id, YNN_NODE_FLAG_KEEP_DIMS));

  // 5. Log of sum.
  uint32_t log_sum_exp_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_unary(
      subgraph, ynn_unary_log, sum_exp_input_minus_max_id, &log_sum_exp_id, 0));

  // 6. Subtract log_sum_exp from (input - max).
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_subtract,
                                        input_minus_max_id, log_sum_exp_id,
                                        &output_id, 0));

  return ynn_status_success;
}

void rms_norm_f32(size_t rows, size_t width, float epsilon, const float* input, const float* weight,
                  const float* residual, bool has_scale, float scale, float* output, size_t row_start, size_t row_end) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
  row_end = std::min(row_end, rows);
  const float reciprocal = 1.0f / static_cast<float>(width);
  for (size_t row = row_start; row < row_end; ++row) {
    const float* x = input + row * width;
    float* y = output + row * width;
    size_t index = 0;
#if defined(__ARM_NEON)
    float32x4_t p0 = vdupq_n_f32(0.0f);
    float32x4_t p1 = vdupq_n_f32(0.0f);
    for (; index + 8 <= width; index += 8) {
        const float32x4_t x0 = vld1q_f32(x + index);
        const float32x4_t x1 = vld1q_f32(x + index + 4);
        p0 = vaddq_f32(p0, vmulq_f32(x0, x0));
        p1 = vaddq_f32(p1, vmulq_f32(x1, x1));
    }
    float partial[8];
    vst1q_f32(partial, p0);
    vst1q_f32(partial + 4, p1);
#else
    float partial[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (; index + 8 <= width; index += 8) {
      for (size_t lane = 0; lane < 8; ++lane) partial[lane] += x[index + lane] * x[index + lane];
    }
#endif
    float sum = 0.0f;
    for (; index < width; ++index) sum += x[index] * x[index];
    for (size_t lane = 0; lane < 8; ++lane) sum += partial[lane];
    const float inverse = 1.0f / std::sqrt(sum * reciprocal + epsilon);
    if (residual == nullptr) {
        index = 0;
#if defined(__ARM_NEON)
        const float32x4_t vinv = vdupq_n_f32(inverse);
        for (; index + 8 <= width; index += 8) {
            vst1q_f32(y + index, vmulq_f32(vmulq_f32(vld1q_f32(x + index), vinv), vld1q_f32(weight + index)));
            vst1q_f32(y + index + 4,
                      vmulq_f32(vmulq_f32(vld1q_f32(x + index + 4), vinv), vld1q_f32(weight + index + 4)));
        }
#endif
        for (; index < width; ++index) y[index] = x[index] * inverse * weight[index];
        continue;
    }
    const float* r = residual + row * width;
    index = 0;
#if defined(__ARM_NEON)
    const float32x4_t vinv = vdupq_n_f32(inverse);
    const float32x4_t vscale = vdupq_n_f32(scale);
    for (; index + 8 <= width; index += 8) {
        float32x4_t v0 = vaddq_f32(vld1q_f32(r + index),
                                   vmulq_f32(vmulq_f32(vld1q_f32(x + index), vinv), vld1q_f32(weight + index)));
        float32x4_t v1 = vaddq_f32(vld1q_f32(r + index + 4),
                                   vmulq_f32(vmulq_f32(vld1q_f32(x + index + 4), vinv), vld1q_f32(weight + index + 4)));
        if (has_scale) {
            v0 = vmulq_f32(v0, vscale);
            v1 = vmulq_f32(v1, vscale);
        }
        vst1q_f32(y + index, v0);
        vst1q_f32(y + index + 4, v1);
    }
#endif
    for (; index < width; ++index) {
        const float value = r[index] + x[index] * inverse * weight[index];
        y[index] = has_scale ? value * scale : value;
    }
  }
}

void rotary_f32(size_t sequence_rows, size_t heads, size_t width, bool shared_angles, const float* input,
                const float* cosine, const float* sine, float* output, size_t row_start, size_t row_end) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
  const size_t half = width / 2;
  for (size_t row = row_start; row < row_end; ++row) {
    const size_t angle = (shared_angles ? row % sequence_rows : row) * half;
    const float* c = cosine + angle;
    const float* s = sine + angle;
    for (size_t head = 0; head < heads; ++head) {
      const float* x = input + (row * heads + head) * width;
      float* y = output + (row * heads + head) * width;
      size_t index = 0;
#if defined(__ARM_NEON)
      for (; index + 4 <= half; index += 4) {
          const float32x4_t first = vld1q_f32(x + index);
          const float32x4_t second = vld1q_f32(x + index + half);
          const float32x4_t vc = vld1q_f32(c + index);
          const float32x4_t vs = vld1q_f32(s + index);
          vst1q_f32(y + index, vsubq_f32(vmulq_f32(first, vc), vmulq_f32(second, vs)));
          vst1q_f32(y + index + half, vaddq_f32(vmulq_f32(second, vc), vmulq_f32(first, vs)));
      }
#endif
      for (; index < half; ++index) {
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
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
  const size_t half = width / 2;
  const float reciprocal = 1.0f / static_cast<float>(width);
  for (size_t vector = vec_start; vector < vec_end; ++vector) {
    const size_t sample = vector / (sequence_rows * heads);
    const size_t position = (vector / heads) % sequence_rows;
    const size_t head = vector % heads;
    const float* x =
        sliced ? input + (sample * input_rows + sequence_offset + position) * input_stride + channel_offset + head * width
               : input + vector * width;
    float* y = output + vector * width;
    size_t index = 0;
#if defined(__ARM_NEON)
    float32x4_t p0 = vdupq_n_f32(0.0f);
    float32x4_t p1 = vdupq_n_f32(0.0f);
    for (; index + 8 <= width; index += 8) {
        const float32x4_t x0 = vld1q_f32(x + index);
        const float32x4_t x1 = vld1q_f32(x + index + 4);
        p0 = vaddq_f32(p0, vmulq_f32(x0, x0));
        p1 = vaddq_f32(p1, vmulq_f32(x1, x1));
    }
    float partial[8];
    vst1q_f32(partial, p0);
    vst1q_f32(partial + 4, p1);
#else
    float partial[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (; index + 8 <= width; index += 8) {
      for (size_t lane = 0; lane < 8; ++lane) partial[lane] += x[index + lane] * x[index + lane];
    }
#endif
    float sum = 0.0f;
    for (; index < width; ++index) sum += x[index] * x[index];
    for (size_t lane = 0; lane < 8; ++lane) sum += partial[lane];
    const float inverse = 1.0f / std::sqrt(sum * reciprocal + epsilon);
    const size_t angle = (shared_angles ? position : sample * sequence_rows + position) * half;
    const float* c = cosine + angle;
    const float* s = sine + angle;
    index = 0;
#if defined(__ARM_NEON)
    const float32x4_t vinv = vdupq_n_f32(inverse);
    for (; index + 4 <= half; index += 4) {
        const float32x4_t first = vmulq_f32(vmulq_f32(vld1q_f32(x + index), vinv), vld1q_f32(weight + index));
        const float32x4_t second =
            vmulq_f32(vmulq_f32(vld1q_f32(x + index + half), vinv), vld1q_f32(weight + index + half));
        const float32x4_t vc = vld1q_f32(c + index);
        const float32x4_t vs = vld1q_f32(s + index);
        vst1q_f32(y + index, vsubq_f32(vmulq_f32(first, vc), vmulq_f32(second, vs)));
        vst1q_f32(y + index + half, vaddq_f32(vmulq_f32(second, vc), vmulq_f32(first, vs)));
    }
#endif
    for (; index < half; ++index) {
        const float first = x[index] * inverse * weight[index];
        const float second = x[index + half] * inverse * weight[index + half];
        y[index] = first * c[index] - second * s[index];
        y[index + half] = second * c[index] + first * s[index];
    }
  }
}

}  // namespace ynn
