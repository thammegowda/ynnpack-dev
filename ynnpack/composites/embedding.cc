// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#if defined(__ARM_NEON) && (defined(__aarch64__) || defined(__arm64__))
#include <arm_neon.h>
#endif

#include "ynnpack/composites/composites.h"

namespace ynn {
namespace {

float fp16_to_fp32_scalar(uint16_t h) {
#if defined(__aarch64__) || defined(__arm64__)
    __fp16 val;
    std::memcpy(&val, &h, sizeof(val));
    return static_cast<float>(val);
#else
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000U) << 16;
    const uint32_t exp = (h >> 10) & 0x1FU;
    const uint32_t mant = h & 0x03FFU;
    if (exp == 0 && mant != 0) {
        const float sub = static_cast<float>(mant) * (1.0f / 16777216.0f);
        return (h & 0x8000U) ? -sub : sub;
    }
    const uint32_t bits =
        exp == 0 ? sign
                 : (exp == 31 ? (sign | 0x7F800000U | (mant << 13)) : (sign | ((exp + 112U) << 23) | (mant << 13)));
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
#endif
}

float bf16_to_fp32_scalar(uint16_t b) {
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

float read_scale_value(const void* base, bool is_fp16, size_t idx) {
  if (!is_fp16) return static_cast<const float*>(base)[idx];
  return fp16_to_fp32_scalar(static_cast<const uint16_t*>(base)[idx]);
}

}  // namespace

void dequantize_embedding_f32(const int32_t* tokens, size_t num_tokens,
                              const void* weights, bool weights_bf16,
                              int32_t packed_bits, bool unsigned_packed,
                              size_t width, size_t groups, const void* scales,
                              bool scales_fp16, const void* zeros,
                              bool zeros_fp16, float multiplier, float* output,
                              size_t row_start, size_t row_end) {
  row_end = std::min(row_end, num_tokens);
  if (packed_bits != 0) {
    const bool has_zeros = zeros != nullptr;
    const size_t group_width = width / groups;
    const size_t per_byte = static_cast<size_t>(8 / packed_bits);
    const auto* data = static_cast<const uint8_t*>(weights);
    const int32_t sign = 1 << (packed_bits - 1);
    const uint8_t sign_mask = unsigned_packed ? (packed_bits == 2   ? 0xAAU
                                                 : packed_bits == 4 ? 0x88U
                                                                    : 0U)
                                              : 0U;
    for (size_t row = row_start; row < row_end; ++row) {
      const size_t token = static_cast<size_t>(tokens[row]);
      for (size_t channel = 0; channel < width; ++channel) {
        const size_t offset = token * width + channel;
        const uint8_t byte =
            static_cast<uint8_t>(data[offset / per_byte] ^ sign_mask);
        const int32_t raw = (byte >> ((offset % per_byte) * packed_bits)) &
                            ((1 << packed_bits) - 1);
        const int32_t integer = (raw ^ sign) - sign;
        const size_t group_index = token * groups + channel / group_width;
        const float sc = read_scale_value(scales, scales_fp16, group_index);
        const float zr =
            has_zeros ? read_scale_value(zeros, zeros_fp16, group_index) : 0.0f;
        output[row * width + channel] =
            (static_cast<float>(integer) * sc + zr) * multiplier;
      }
    }
    return;
  }

  if (weights_bf16) {
    const auto* w_u16 = static_cast<const uint16_t*>(weights);
    for (size_t row = row_start; row < row_end; ++row) {
      const size_t base = static_cast<size_t>(tokens[row]) * width;
      for (size_t channel = 0; channel < width; ++channel) {
        output[row * width + channel] =
            bf16_to_fp32_scalar(w_u16[base + channel]) * multiplier;
      }
    }
    return;
  }

  const auto* w_f32 = static_cast<const float*>(weights);
  for (size_t row = row_start; row < row_end; ++row) {
    const size_t base = static_cast<size_t>(tokens[row]) * width;
    for (size_t channel = 0; channel < width; ++channel) {
      output[row * width + channel] = w_f32[base + channel] * multiplier;
    }
  }
}

void quantize_symmetric_int8_f32(size_t count, float scale, const float* input,
                                 int8_t* output) {
    size_t index = 0;
#if defined(__ARM_NEON) && (defined(__aarch64__) || defined(__arm64__))
    const float32x4_t vscale = vdupq_n_f32(scale);
    for (; index + 16 <= count; index += 16) {
        const int32x4_t i0 = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(input + index + 0), vscale));
        const int32x4_t i1 = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(input + index + 4), vscale));
        const int32x4_t i2 = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(input + index + 8), vscale));
        const int32x4_t i3 = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(input + index + 12), vscale));
        const int16x8_t s0 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
        const int16x8_t s1 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
        vst1q_s8(output + index, vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1)));
    }
#endif
    for (; index < count; ++index) {
        const float rounded = std::nearbyint(input[index] / scale);
        output[index] = static_cast<int8_t>(std::clamp(rounded, -128.0f, 127.0f));
    }
}

void argmax_chunk_f32(size_t count, const float* row_logits, size_t col_offset, float* out_max, int32_t* out_index,
                      bool* out_invalid) {
    float best = -std::numeric_limits<float>::infinity();
    int32_t token = -1;
    bool invalid = false;
    size_t column = 0;
#if defined(__ARM_NEON) && (defined(__aarch64__) || defined(__arm64__))
    uint32x4_t nan_or_pinf = vdupq_n_u32(0);
    const uint32x4_t exp_mask = vdupq_n_u32(0x7F800000u);
    const uint32x4_t ninf_bits = vdupq_n_u32(0xFF800000u);
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
#endif
    for (; column < count; ++column) {
        const float value = row_logits[column];
        invalid |= std::isnan(value) || value == std::numeric_limits<float>::infinity();
        if (value > best) {
            best = value;
            token = static_cast<int32_t>(col_offset + column);
        }
    }
    *out_max = best;
    *out_index = token;
    *out_invalid = invalid;
}

void argmax_rows_f32(size_t rows, size_t width, const float* logits,
                     int32_t* tokens) {
  for (size_t row = 0; row < rows; ++row) {
    float best = -std::numeric_limits<float>::infinity();
    int32_t token = -1;
    bool invalid = false;
    argmax_chunk_f32(width, logits + row * width, 0, &best, &token, &invalid);
    tokens[row] = invalid ? -1 : token;
  }
}

}  // namespace ynn
