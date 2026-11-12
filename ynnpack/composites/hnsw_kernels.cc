// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ynnpack/base/arch.h"
#include "ynnpack/composites/hnsw.h"

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define YNN_HNSW_AVX512 1
#endif

#if defined(__ARM_NEON) && (defined(__aarch64__) || defined(__arm64__))
#include <arm_neon.h>
#define YNN_HNSW_NEON 1
#if defined(__clang__)
#define YNN_HNSW_TARGET_BF16 __attribute__((target("bf16")))
#define YNN_HNSW_TARGET_DOTPROD __attribute__((target("dotprod")))
#define YNN_HNSW_TARGET_DOTPROD_FP16 __attribute__((target("dotprod,fp16")))
#define YNN_HNSW_ARM64_TARGETS 1
#elif defined(__GNUC__)
#define YNN_HNSW_TARGET_BF16 __attribute__((target("+bf16")))
#define YNN_HNSW_TARGET_DOTPROD __attribute__((target("+dotprod")))
#define YNN_HNSW_TARGET_DOTPROD_FP16 __attribute__((target("+dotprod+fp16")))
#define YNN_HNSW_ARM64_TARGETS 1
#endif
#endif

namespace ynn {
namespace {

inline float bit_cast_f32(uint32_t bits) noexcept {
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

template <bool BFloat>
inline float weight_elem(const void* row, size_t column) noexcept {
  if constexpr (BFloat) {
    const uint32_t bits =
        static_cast<uint32_t>(static_cast<const uint16_t*>(row)[column]) << 16;
    return bit_cast_f32(bits);
  } else {
    return static_cast<const float*>(row)[column];
  }
}

template <bool BFloat>
float dot_portable(const float* query, const void* row, size_t width) {
  std::array<float, 16> lanes{};
  size_t column = 0;
  for (; column + 16 <= width; column += 16) {
    for (size_t lane = 0; lane < 16; ++lane) {
      lanes[lane] = std::fma(query[column + lane],
                             weight_elem<BFloat>(row, column + lane),
                             lanes[lane]);
    }
  }
  for (size_t offset = 8; offset != 0; offset /= 2) {
    for (size_t lane = 0; lane < offset; ++lane) {
      lanes[lane] += lanes[lane + offset];
    }
  }
  float tail = 0.0f;
  for (; column < width; ++column) {
    tail = std::fma(query[column], weight_elem<BFloat>(row, column), tail);
  }
  return lanes[0] + tail;
}

template <bool BFloat>
void dot4_portable(const float* query, const void* r0, const void* r1,
                   const void* r2, const void* r3, size_t width,
                   float out[4]) {
  out[0] = dot_portable<BFloat>(query, r0, width);
  out[1] = dot_portable<BFloat>(query, r1, width);
  out[2] = dot_portable<BFloat>(query, r2, width);
  out[3] = dot_portable<BFloat>(query, r3, width);
}

template <bool BFloat>
float dot_rows_portable(const void* left, const void* right, size_t width) {
  std::array<float, 16> lanes{};
  size_t column = 0;
  for (; column + 16 <= width; column += 16) {
    for (size_t lane = 0; lane < 16; ++lane) {
      lanes[lane] = std::fma(weight_elem<BFloat>(left, column + lane),
                             weight_elem<BFloat>(right, column + lane),
                             lanes[lane]);
    }
  }
  for (size_t offset = 8; offset != 0; offset /= 2) {
    for (size_t lane = 0; lane < offset; ++lane) {
      lanes[lane] += lanes[lane + offset];
    }
  }
  float tail = 0.0f;
  for (; column < width; ++column) {
    tail = std::fma(weight_elem<BFloat>(left, column),
                    weight_elem<BFloat>(right, column), tail);
  }
  return lanes[0] + tail;
}

template <bool BFloat>
void dot_rows4_portable(const void* query, const void* r0, const void* r1,
                        const void* r2, const void* r3, size_t width,
                        float out[4]) {
  out[0] = dot_rows_portable<BFloat>(query, r0, width);
  out[1] = dot_rows_portable<BFloat>(query, r1, width);
  out[2] = dot_rows_portable<BFloat>(query, r2, width);
  out[3] = dot_rows_portable<BFloat>(query, r3, width);
}

#if defined(YNN_HNSW_AVX512)
template <bool BFloat>
__attribute__((target("avx512f,avx512bw,fma"))) float dot_avx512(
    const float* query, const void* row, size_t width) {
  __m512 sum = _mm512_setzero_ps();
  const auto* bytes = static_cast<const uint8_t*>(row);
  size_t column = 0;
  for (; column + 16 <= width; column += 16) {
    __m512 values;
    if constexpr (BFloat) {
      const __m256i packed = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(bytes + column * sizeof(uint16_t)));
      values = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
    } else {
      values = _mm512_loadu_ps(static_cast<const float*>(row) + column);
    }
    sum = _mm512_fmadd_ps(_mm512_loadu_ps(query + column), values, sum);
  }
  float tail = 0.0f;
  for (; column < width; ++column) {
    tail = std::fma(query[column], weight_elem<BFloat>(row, column), tail);
  }
  return _mm512_reduce_add_ps(sum) + tail;
}

template <bool BFloat>
__attribute__((target("avx512f,avx512bw,fma"))) void dot4_avx512(
    const float* query, const void* r0, const void* r1, const void* r2,
    const void* r3, size_t width, float out[4]) {
  __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
  __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
  const auto load = [](const void* row, size_t col) {
    if constexpr (BFloat) {
      const auto* bytes = static_cast<const uint8_t*>(row);
      const __m256i packed = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(bytes + col * sizeof(uint16_t)));
      return _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
    } else {
      return _mm512_loadu_ps(static_cast<const float*>(row) + col);
    }
  };
  size_t column = 0;
  for (; column + 16 <= width; column += 16) {
    const __m512 q = _mm512_loadu_ps(query + column);
    s0 = _mm512_fmadd_ps(q, load(r0, column), s0);
    s1 = _mm512_fmadd_ps(q, load(r1, column), s1);
    s2 = _mm512_fmadd_ps(q, load(r2, column), s2);
    s3 = _mm512_fmadd_ps(q, load(r3, column), s3);
  }
  out[0] = _mm512_reduce_add_ps(s0);
  out[1] = _mm512_reduce_add_ps(s1);
  out[2] = _mm512_reduce_add_ps(s2);
  out[3] = _mm512_reduce_add_ps(s3);
  for (; column < width; ++column) {
    const float q = query[column];
    out[0] = std::fma(q, weight_elem<BFloat>(r0, column), out[0]);
    out[1] = std::fma(q, weight_elem<BFloat>(r1, column), out[1]);
    out[2] = std::fma(q, weight_elem<BFloat>(r2, column), out[2]);
    out[3] = std::fma(q, weight_elem<BFloat>(r3, column), out[3]);
  }
}

template <bool BFloat>
__attribute__((target("avx512f,avx512bw,fma"))) float dot_rows_avx512(
    const void* left, const void* right, size_t width) {
  __m512 sum = _mm512_setzero_ps();
  const auto* l_bytes = static_cast<const uint8_t*>(left);
  const auto* r_bytes = static_cast<const uint8_t*>(right);
  size_t column = 0;
  for (; column + 16 <= width; column += 16) {
    __m512 first, second;
    if constexpr (BFloat) {
      const __m256i l_packed = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(l_bytes + column * sizeof(uint16_t)));
      const __m256i r_packed = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(r_bytes + column * sizeof(uint16_t)));
      first = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(l_packed), 16));
      second = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(r_packed), 16));
    } else {
      first = _mm512_loadu_ps(static_cast<const float*>(left) + column);
      second = _mm512_loadu_ps(static_cast<const float*>(right) + column);
    }
    sum = _mm512_fmadd_ps(first, second, sum);
  }
  float tail = 0.0f;
  for (; column < width; ++column) {
    tail = std::fma(weight_elem<BFloat>(left, column),
                    weight_elem<BFloat>(right, column), tail);
  }
  return _mm512_reduce_add_ps(sum) + tail;
}

template <bool BFloat>
__attribute__((target("avx512f,avx512bw,fma"))) void dot_rows4_avx512(
    const void* query, const void* r0, const void* r1, const void* r2,
    const void* r3, size_t width, float out[4]) {
  __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
  __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
  const auto load = [](const void* row, size_t col) {
    if constexpr (BFloat) {
      const auto* bytes = static_cast<const uint8_t*>(row);
      const __m256i packed = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(bytes + col * sizeof(uint16_t)));
      return _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
    } else {
      return _mm512_loadu_ps(static_cast<const float*>(row) + col);
    }
  };
  size_t column = 0;
  for (; column + 16 <= width; column += 16) {
    const __m512 q = load(query, column);
    s0 = _mm512_fmadd_ps(q, load(r0, column), s0);
    s1 = _mm512_fmadd_ps(q, load(r1, column), s1);
    s2 = _mm512_fmadd_ps(q, load(r2, column), s2);
    s3 = _mm512_fmadd_ps(q, load(r3, column), s3);
  }
  out[0] = _mm512_reduce_add_ps(s0);
  out[1] = _mm512_reduce_add_ps(s1);
  out[2] = _mm512_reduce_add_ps(s2);
  out[3] = _mm512_reduce_add_ps(s3);
  for (; column < width; ++column) {
    const float q = weight_elem<BFloat>(query, column);
    out[0] = std::fma(q, weight_elem<BFloat>(r0, column), out[0]);
    out[1] = std::fma(q, weight_elem<BFloat>(r1, column), out[1]);
    out[2] = std::fma(q, weight_elem<BFloat>(r2, column), out[2]);
    out[3] = std::fma(q, weight_elem<BFloat>(r3, column), out[3]);
  }
}
#endif

#if defined(YNN_HNSW_NEON)
template <bool BFloat>
float dot_neon(const float* query, const void* row, size_t width) {
  float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
  float32x4_t s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);
  size_t column = 0;
  for (; column + 16 <= width; column += 16) {
    const float32x4_t q0 = vld1q_f32(query + column);
    const float32x4_t q1 = vld1q_f32(query + column + 4);
    const float32x4_t q2 = vld1q_f32(query + column + 8);
    const float32x4_t q3 = vld1q_f32(query + column + 12);
    if constexpr (BFloat) {
      const auto* w_u16 = static_cast<const uint16_t*>(row) + column;
      const uint16x8_t lo = vld1q_u16(w_u16);
      const uint16x8_t hi = vld1q_u16(w_u16 + 8);
      s0 = vfmaq_f32(s0, q0,
                     vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(lo), 16)));
      s1 = vfmaq_f32(s1, q1,
                     vreinterpretq_f32_u32(vshll_high_n_u16(lo, 16)));
      s2 = vfmaq_f32(s2, q2,
                     vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(hi), 16)));
      s3 = vfmaq_f32(s3, q3,
                     vreinterpretq_f32_u32(vshll_high_n_u16(hi, 16)));
    } else {
      const auto* w_f32 = static_cast<const float*>(row) + column;
      s0 = vfmaq_f32(s0, q0, vld1q_f32(w_f32));
      s1 = vfmaq_f32(s1, q1, vld1q_f32(w_f32 + 4));
      s2 = vfmaq_f32(s2, q2, vld1q_f32(w_f32 + 8));
      s3 = vfmaq_f32(s3, q3, vld1q_f32(w_f32 + 12));
    }
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; column < width; ++column) {
    sum = std::fma(query[column], weight_elem<BFloat>(row, column), sum);
  }
  return sum;
}

template <bool BFloat>
void dot4_neon(const float* query, const void* r0, const void* r1,
               const void* r2, const void* r3, size_t width, float out[4]) {
  float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
  float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
  size_t column = 0;
  for (; column + 8 <= width; column += 8) {
    const float32x4_t q0 = vld1q_f32(query + column);
    const float32x4_t q1 = vld1q_f32(query + column + 4);
    if constexpr (BFloat) {
      const uint16x8_t v0 = vld1q_u16(static_cast<const uint16_t*>(r0) + column);
      const uint16x8_t v1 = vld1q_u16(static_cast<const uint16_t*>(r1) + column);
      const uint16x8_t v2 = vld1q_u16(static_cast<const uint16_t*>(r2) + column);
      const uint16x8_t v3 = vld1q_u16(static_cast<const uint16_t*>(r3) + column);
      a0 = vfmaq_f32(
          vfmaq_f32(a0, q0,
                    vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v0), 16))),
          q1, vreinterpretq_f32_u32(vshll_high_n_u16(v0, 16)));
      a1 = vfmaq_f32(
          vfmaq_f32(a1, q0,
                    vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v1), 16))),
          q1, vreinterpretq_f32_u32(vshll_high_n_u16(v1, 16)));
      a2 = vfmaq_f32(
          vfmaq_f32(a2, q0,
                    vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v2), 16))),
          q1, vreinterpretq_f32_u32(vshll_high_n_u16(v2, 16)));
      a3 = vfmaq_f32(
          vfmaq_f32(a3, q0,
                    vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v3), 16))),
          q1, vreinterpretq_f32_u32(vshll_high_n_u16(v3, 16)));
    } else {
      const auto* f0 = static_cast<const float*>(r0) + column;
      const auto* f1 = static_cast<const float*>(r1) + column;
      const auto* f2 = static_cast<const float*>(r2) + column;
      const auto* f3 = static_cast<const float*>(r3) + column;
      a0 = vfmaq_f32(vfmaq_f32(a0, q0, vld1q_f32(f0)), q1, vld1q_f32(f0 + 4));
      a1 = vfmaq_f32(vfmaq_f32(a1, q0, vld1q_f32(f1)), q1, vld1q_f32(f1 + 4));
      a2 = vfmaq_f32(vfmaq_f32(a2, q0, vld1q_f32(f2)), q1, vld1q_f32(f2 + 4));
      a3 = vfmaq_f32(vfmaq_f32(a3, q0, vld1q_f32(f3)), q1, vld1q_f32(f3 + 4));
    }
  }
  out[0] = vaddvq_f32(a0);
  out[1] = vaddvq_f32(a1);
  out[2] = vaddvq_f32(a2);
  out[3] = vaddvq_f32(a3);
  for (; column < width; ++column) {
    const float q = query[column];
    out[0] = std::fma(q, weight_elem<BFloat>(r0, column), out[0]);
    out[1] = std::fma(q, weight_elem<BFloat>(r1, column), out[1]);
    out[2] = std::fma(q, weight_elem<BFloat>(r2, column), out[2]);
    out[3] = std::fma(q, weight_elem<BFloat>(r3, column), out[3]);
  }
}

template <bool BFloat>
float dot_rows_neon(const void* left, const void* right, size_t width) {
  float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
  float32x4_t s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);
  size_t column = 0;
  for (; column + 16 <= width; column += 16) {
    if constexpr (BFloat) {
      const auto* l_u16 = static_cast<const uint16_t*>(left) + column;
      const auto* r_u16 = static_cast<const uint16_t*>(right) + column;
      const uint16x8_t l_lo = vld1q_u16(l_u16), l_hi = vld1q_u16(l_u16 + 8);
      const uint16x8_t r_lo = vld1q_u16(r_u16), r_hi = vld1q_u16(r_u16 + 8);
      s0 = vfmaq_f32(
          s0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(l_lo), 16)),
          vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r_lo), 16)));
      s1 = vfmaq_f32(s1, vreinterpretq_f32_u32(vshll_high_n_u16(l_lo, 16)),
                     vreinterpretq_f32_u32(vshll_high_n_u16(r_lo, 16)));
      s2 = vfmaq_f32(
          s2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(l_hi), 16)),
          vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r_hi), 16)));
      s3 = vfmaq_f32(s3, vreinterpretq_f32_u32(vshll_high_n_u16(l_hi, 16)),
                     vreinterpretq_f32_u32(vshll_high_n_u16(r_hi, 16)));
    } else {
      const auto* l_f32 = static_cast<const float*>(left) + column;
      const auto* r_f32 = static_cast<const float*>(right) + column;
      s0 = vfmaq_f32(s0, vld1q_f32(l_f32), vld1q_f32(r_f32));
      s1 = vfmaq_f32(s1, vld1q_f32(l_f32 + 4), vld1q_f32(r_f32 + 4));
      s2 = vfmaq_f32(s2, vld1q_f32(l_f32 + 8), vld1q_f32(r_f32 + 8));
      s3 = vfmaq_f32(s3, vld1q_f32(l_f32 + 12), vld1q_f32(r_f32 + 12));
    }
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; column < width; ++column) {
    sum = std::fma(weight_elem<BFloat>(left, column),
                   weight_elem<BFloat>(right, column), sum);
  }
  return sum;
}

template <bool BFloat>
void dot_rows4_neon(const void* query, const void* r0, const void* r1,
                    const void* r2, const void* r3, size_t width,
                    float out[4]) {
  float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
  float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
  size_t column = 0;
  for (; column + 8 <= width; column += 8) {
    if constexpr (BFloat) {
      const uint16x8_t qv =
          vld1q_u16(static_cast<const uint16_t*>(query) + column);
      const float32x4_t q0 =
          vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(qv), 16));
      const float32x4_t q1 = vreinterpretq_f32_u32(vshll_high_n_u16(qv, 16));
      const uint16x8_t v0 = vld1q_u16(static_cast<const uint16_t*>(r0) + column);
      const uint16x8_t v1 = vld1q_u16(static_cast<const uint16_t*>(r1) + column);
      const uint16x8_t v2 = vld1q_u16(static_cast<const uint16_t*>(r2) + column);
      const uint16x8_t v3 = vld1q_u16(static_cast<const uint16_t*>(r3) + column);
      a0 = vfmaq_f32(
          vfmaq_f32(a0, q0,
                    vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v0), 16))),
          q1, vreinterpretq_f32_u32(vshll_high_n_u16(v0, 16)));
      a1 = vfmaq_f32(
          vfmaq_f32(a1, q0,
                    vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v1), 16))),
          q1, vreinterpretq_f32_u32(vshll_high_n_u16(v1, 16)));
      a2 = vfmaq_f32(
          vfmaq_f32(a2, q0,
                    vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v2), 16))),
          q1, vreinterpretq_f32_u32(vshll_high_n_u16(v2, 16)));
      a3 = vfmaq_f32(
          vfmaq_f32(a3, q0,
                    vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v3), 16))),
          q1, vreinterpretq_f32_u32(vshll_high_n_u16(v3, 16)));
    } else {
      const auto* fq = static_cast<const float*>(query) + column;
      const float32x4_t q0 = vld1q_f32(fq);
      const float32x4_t q1 = vld1q_f32(fq + 4);
      const auto* f0 = static_cast<const float*>(r0) + column;
      const auto* f1 = static_cast<const float*>(r1) + column;
      const auto* f2 = static_cast<const float*>(r2) + column;
      const auto* f3 = static_cast<const float*>(r3) + column;
      a0 = vfmaq_f32(vfmaq_f32(a0, q0, vld1q_f32(f0)), q1, vld1q_f32(f0 + 4));
      a1 = vfmaq_f32(vfmaq_f32(a1, q0, vld1q_f32(f1)), q1, vld1q_f32(f1 + 4));
      a2 = vfmaq_f32(vfmaq_f32(a2, q0, vld1q_f32(f2)), q1, vld1q_f32(f2 + 4));
      a3 = vfmaq_f32(vfmaq_f32(a3, q0, vld1q_f32(f3)), q1, vld1q_f32(f3 + 4));
    }
  }
  out[0] = vaddvq_f32(a0);
  out[1] = vaddvq_f32(a1);
  out[2] = vaddvq_f32(a2);
  out[3] = vaddvq_f32(a3);
  for (; column < width; ++column) {
    const float qv = weight_elem<BFloat>(query, column);
    out[0] = std::fma(qv, weight_elem<BFloat>(r0, column), out[0]);
    out[1] = std::fma(qv, weight_elem<BFloat>(r1, column), out[1]);
    out[2] = std::fma(qv, weight_elem<BFloat>(r2, column), out[2]);
    out[3] = std::fma(qv, weight_elem<BFloat>(r3, column), out[3]);
  }
}

#if defined(YNN_HNSW_ARM64_TARGETS)
YNN_HNSW_TARGET_BF16 float dot_rows_bf16_neon(const void* left,
                                              const void* right, size_t width) {
  float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
  float32x4_t s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);
  const auto* l_u16 = static_cast<const uint16_t*>(left);
  const auto* r_u16 = static_cast<const uint16_t*>(right);
  size_t column = 0;
  for (; column + 32 <= width; column += 32) {
    s0 = vbfdotq_f32(s0, vreinterpretq_bf16_u16(vld1q_u16(l_u16 + column)),
                     vreinterpretq_bf16_u16(vld1q_u16(r_u16 + column)));
    s1 = vbfdotq_f32(s1, vreinterpretq_bf16_u16(vld1q_u16(l_u16 + column + 8)),
                     vreinterpretq_bf16_u16(vld1q_u16(r_u16 + column + 8)));
    s2 = vbfdotq_f32(s2, vreinterpretq_bf16_u16(vld1q_u16(l_u16 + column + 16)),
                     vreinterpretq_bf16_u16(vld1q_u16(r_u16 + column + 16)));
    s3 = vbfdotq_f32(s3, vreinterpretq_bf16_u16(vld1q_u16(l_u16 + column + 24)),
                     vreinterpretq_bf16_u16(vld1q_u16(r_u16 + column + 24)));
  }
  for (; column + 8 <= width; column += 8) {
    s0 = vbfdotq_f32(s0, vreinterpretq_bf16_u16(vld1q_u16(l_u16 + column)),
                     vreinterpretq_bf16_u16(vld1q_u16(r_u16 + column)));
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; column < width; ++column) {
    sum = std::fma(weight_elem<true>(left, column),
                   weight_elem<true>(right, column), sum);
  }
  return sum;
}

YNN_HNSW_TARGET_BF16 void dot_rows4_bf16_neon(const void* query, const void* r0,
                                              const void* r1, const void* r2,
                                              const void* r3, size_t width,
                                              float out[4]) {
  float32x4_t a0 = vdupq_n_f32(0.0f), b0 = vdupq_n_f32(0.0f);
  float32x4_t a1 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f);
  float32x4_t a2 = vdupq_n_f32(0.0f), b2 = vdupq_n_f32(0.0f);
  float32x4_t a3 = vdupq_n_f32(0.0f), b3 = vdupq_n_f32(0.0f);
  const auto* q = static_cast<const uint16_t*>(query);
  const auto* p0 = static_cast<const uint16_t*>(r0);
  const auto* p1 = static_cast<const uint16_t*>(r1);
  const auto* p2 = static_cast<const uint16_t*>(r2);
  const auto* p3 = static_cast<const uint16_t*>(r3);
  size_t column = 0;
  for (; column + 16 <= width; column += 16) {
    const bfloat16x8_t q0 = vreinterpretq_bf16_u16(vld1q_u16(q + column));
    const bfloat16x8_t q1 = vreinterpretq_bf16_u16(vld1q_u16(q + column + 8));
    a0 = vbfdotq_f32(a0, q0, vreinterpretq_bf16_u16(vld1q_u16(p0 + column)));
    b0 = vbfdotq_f32(b0, q1,
                     vreinterpretq_bf16_u16(vld1q_u16(p0 + column + 8)));
    a1 = vbfdotq_f32(a1, q0, vreinterpretq_bf16_u16(vld1q_u16(p1 + column)));
    b1 = vbfdotq_f32(b1, q1,
                     vreinterpretq_bf16_u16(vld1q_u16(p1 + column + 8)));
    a2 = vbfdotq_f32(a2, q0, vreinterpretq_bf16_u16(vld1q_u16(p2 + column)));
    b2 = vbfdotq_f32(b2, q1,
                     vreinterpretq_bf16_u16(vld1q_u16(p2 + column + 8)));
    a3 = vbfdotq_f32(a3, q0, vreinterpretq_bf16_u16(vld1q_u16(p3 + column)));
    b3 = vbfdotq_f32(b3, q1,
                     vreinterpretq_bf16_u16(vld1q_u16(p3 + column + 8)));
  }
  for (; column + 8 <= width; column += 8) {
    const bfloat16x8_t q0 = vreinterpretq_bf16_u16(vld1q_u16(q + column));
    a0 = vbfdotq_f32(a0, q0, vreinterpretq_bf16_u16(vld1q_u16(p0 + column)));
    a1 = vbfdotq_f32(a1, q0, vreinterpretq_bf16_u16(vld1q_u16(p1 + column)));
    a2 = vbfdotq_f32(a2, q0, vreinterpretq_bf16_u16(vld1q_u16(p2 + column)));
    a3 = vbfdotq_f32(a3, q0, vreinterpretq_bf16_u16(vld1q_u16(p3 + column)));
  }
  out[0] = vaddvq_f32(vaddq_f32(a0, b0));
  out[1] = vaddvq_f32(vaddq_f32(a1, b1));
  out[2] = vaddvq_f32(vaddq_f32(a2, b2));
  out[3] = vaddvq_f32(vaddq_f32(a3, b3));
  for (; column < width; ++column) {
    const float qv = weight_elem<true>(query, column);
    out[0] = std::fma(qv, weight_elem<true>(r0, column), out[0]);
    out[1] = std::fma(qv, weight_elem<true>(r1, column), out[1]);
    out[2] = std::fma(qv, weight_elem<true>(r2, column), out[2]);
    out[3] = std::fma(qv, weight_elem<true>(r3, column), out[3]);
  }
}
#endif
#endif

inline bool has_arm64_bf16() noexcept {
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) && \
    defined(YNN_HNSW_ARM64_TARGETS)
  return true;
#elif defined(YNN_HNSW_ARM64_TARGETS) && defined(YNN_ARCH_ARM)
  return is_arch_supported(arch_flag::neonbf16);
#else
  return false;
#endif
}

inline bool has_arm64_dotprod() noexcept {
#if defined(__ARM_FEATURE_DOTPROD) && defined(YNN_HNSW_ARM64_TARGETS)
  return true;
#elif defined(YNN_HNSW_ARM64_TARGETS) && defined(YNN_ARCH_ARM)
  return is_arch_supported(arch_flag::neondot);
#else
  return false;
#endif
}

inline bool has_arm64_dotprod_fp16() noexcept {
#if defined(__ARM_FEATURE_DOTPROD) && \
    defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && \
    defined(YNN_HNSW_ARM64_TARGETS)
  return true;
#elif defined(YNN_HNSW_ARM64_TARGETS) && defined(YNN_ARCH_ARM)
  return is_arch_supported(arch_flag::neondot | arch_flag::neonfp16arith);
#else
  return false;
#endif
}

float read_meta(const uint8_t* base, bool f16, size_t idx) noexcept {
  if (f16) {
#if defined(YNN_HNSW_NEON)
    __fp16 val;
    std::memcpy(&val, base + idx * sizeof(uint16_t), sizeof(val));
    return static_cast<float>(val);
#else
    uint16_t h = 0;
    std::memcpy(&h, base + idx * sizeof(uint16_t), sizeof(h));
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x03ffu;
    if (exp == 0) {
      return std::copysign(static_cast<float>(mant) * (1.0f / 16777216.0f),
                           bit_cast_f32(sign));
    }
    if (exp == 31) return bit_cast_f32(sign | 0x7f800000u | (mant << 13));
    return bit_cast_f32(sign | ((exp + 112u) << 23) | (mant << 13));
#endif
  }
  float out = 0.0f;
  std::memcpy(&out, base + idx * sizeof(float), sizeof(out));
  return out;
}

#if defined(YNN_HNSW_ARM64_TARGETS)
template <bool ScaleF16, bool HasZero, bool ZeroF16>
YNN_HNSW_TARGET_DOTPROD_FP16 float quantized_slice_dot_int4_neon(
    const uint8_t* p0, size_t B, size_t chunks, const int8_t* q_i8,
    const float* q_scale, const float* q_bsum) noexcept {
  float32x4_t f0 = vdupq_n_f32(0.0f);
  const size_t words_per_block = chunks * 4;
  const size_t w_step = words_per_block * 16;
  constexpr size_t s_step = ScaleF16 ? 8 : 16;
  constexpr size_t z_step = HasZero ? (ZeroF16 ? 8 : 16) : 0;
  for (size_t b = 0; b < B; b += 4) {
    int32x4_t d0_e = vdupq_n_s32(0), d0_o = vdupq_n_s32(0);
    for (size_t k = 0; k < words_per_block; ++k, q_i8 += 32) {
      const uint8x16_t r0 = vld1q_u8(p0 + k * 16);
      d0_o = vdotq_s32(d0_o, vshrq_n_s8(vreinterpretq_s8_u8(r0), 4),
                       vld1q_s8(q_i8 + 16));
      d0_e = vdotq_s32(
          d0_e, vshrq_n_s8(vreinterpretq_s8_u8(vshlq_n_u8(r0, 4)), 4),
          vld1q_s8(q_i8));
    }
    p0 += w_step;
    const int32x4_t d0 = vaddq_s32(d0_e, d0_o);
    const float32x4_t qs = vld1q_f32(q_scale + b);
    if constexpr (ScaleF16 && HasZero && ZeroF16) {
      const float16x8_t sz0 =
          vld1q_f16(reinterpret_cast<const __fp16*>(p0));
      f0 = vfmaq_f32(vfmaq_f32(f0, vcvtq_f32_s32(d0),
                               vmulq_f32(vcvt_f32_f16(vget_low_f16(sz0)), qs)),
                     vcvt_high_f32_f16(sz0), vld1q_f32(q_bsum + b));
    } else {
      const float32x4_t ws =
          ScaleF16 ? vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p0)))
                   : vld1q_f32(reinterpret_cast<const float*>(p0));
      f0 = vfmaq_f32(f0, vcvtq_f32_s32(d0), vmulq_f32(ws, qs));
      if constexpr (HasZero) {
        const uint8_t* zp = p0 + s_step;
        const float32x4_t wz =
            ZeroF16
                ? vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(zp)))
                : vld1q_f32(reinterpret_cast<const float*>(zp));
        f0 = vfmaq_f32(f0, wz, vld1q_f32(q_bsum + b));
      }
    }
    p0 += s_step + z_step;
  }
  return vaddvq_f32(f0);
}

template <bool ScaleF16, bool HasZero, bool ZeroF16>
YNN_HNSW_TARGET_DOTPROD_FP16 void quantized_slices4_dot_int4_neon(
    const uint8_t* p0, const uint8_t* p1, const uint8_t* p2, const uint8_t* p3,
    size_t B, size_t chunks, const int8_t* q_i8, const float* q_scale,
    const float* q_bsum, float out[4]) noexcept {
  float32x4_t f0 = vdupq_n_f32(0.0f), f1 = vdupq_n_f32(0.0f);
  float32x4_t f2 = vdupq_n_f32(0.0f), f3 = vdupq_n_f32(0.0f);
  const size_t words_per_block = chunks * 4;
  const size_t w_step = words_per_block * 16;
  constexpr size_t s_step = ScaleF16 ? 8 : 16;
  constexpr size_t z_step = HasZero ? (ZeroF16 ? 8 : 16) : 0;
  for (size_t b = 0; b < B; b += 4) {
    int32x4_t d0 = vdupq_n_s32(0), d1 = vdupq_n_s32(0);
    int32x4_t d2 = vdupq_n_s32(0), d3 = vdupq_n_s32(0);
    for (size_t k = 0; k < words_per_block; ++k, q_i8 += 32) {
      const int8x16_t qe = vld1q_s8(q_i8);
      const int8x16_t qo = vld1q_s8(q_i8 + 16);

      const uint8x16_t r0 = vld1q_u8(p0 + k * 16);
      const uint8x16_t r1 = vld1q_u8(p1 + k * 16);
      const uint8x16_t r2 = vld1q_u8(p2 + k * 16);
      const uint8x16_t r3 = vld1q_u8(p3 + k * 16);

      d0 = vdotq_s32(d0, vshrq_n_s8(vreinterpretq_s8_u8(r0), 4), qo);
      d1 = vdotq_s32(d1, vshrq_n_s8(vreinterpretq_s8_u8(r1), 4), qo);
      d2 = vdotq_s32(d2, vshrq_n_s8(vreinterpretq_s8_u8(r2), 4), qo);
      d3 = vdotq_s32(d3, vshrq_n_s8(vreinterpretq_s8_u8(r3), 4), qo);

      d0 = vdotq_s32(
          d0, vshrq_n_s8(vreinterpretq_s8_u8(vshlq_n_u8(r0, 4)), 4), qe);
      d1 = vdotq_s32(
          d1, vshrq_n_s8(vreinterpretq_s8_u8(vshlq_n_u8(r1, 4)), 4), qe);
      d2 = vdotq_s32(
          d2, vshrq_n_s8(vreinterpretq_s8_u8(vshlq_n_u8(r2, 4)), 4), qe);
      d3 = vdotq_s32(
          d3, vshrq_n_s8(vreinterpretq_s8_u8(vshlq_n_u8(r3, 4)), 4), qe);
    }
    p0 += w_step;
    p1 += w_step;
    p2 += w_step;
    p3 += w_step;
    const float32x4_t qs = vld1q_f32(q_scale + b);
    if constexpr (ScaleF16 && HasZero && ZeroF16) {
      const float16x8_t sz0 = vld1q_f16(reinterpret_cast<const __fp16*>(p0));
      const float16x8_t sz1 = vld1q_f16(reinterpret_cast<const __fp16*>(p1));
      const float16x8_t sz2 = vld1q_f16(reinterpret_cast<const __fp16*>(p2));
      const float16x8_t sz3 = vld1q_f16(reinterpret_cast<const __fp16*>(p3));
      const float32x4_t qz = vld1q_f32(q_bsum + b);
      f0 = vfmaq_f32(vfmaq_f32(f0, vcvtq_f32_s32(d0),
                               vmulq_f32(vcvt_f32_f16(vget_low_f16(sz0)), qs)),
                     vcvt_high_f32_f16(sz0), qz);
      f1 = vfmaq_f32(vfmaq_f32(f1, vcvtq_f32_s32(d1),
                               vmulq_f32(vcvt_f32_f16(vget_low_f16(sz1)), qs)),
                     vcvt_high_f32_f16(sz1), qz);
      f2 = vfmaq_f32(vfmaq_f32(f2, vcvtq_f32_s32(d2),
                               vmulq_f32(vcvt_f32_f16(vget_low_f16(sz2)), qs)),
                     vcvt_high_f32_f16(sz2), qz);
      f3 = vfmaq_f32(vfmaq_f32(f3, vcvtq_f32_s32(d3),
                               vmulq_f32(vcvt_f32_f16(vget_low_f16(sz3)), qs)),
                     vcvt_high_f32_f16(sz3), qz);
    } else {
      const auto load_scale = [](const uint8_t* base) noexcept -> float32x4_t {
        if constexpr (ScaleF16) {
          return vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(base)));
        } else {
          return vld1q_f32(reinterpret_cast<const float*>(base));
        }
      };
      f0 = vfmaq_f32(f0, vcvtq_f32_s32(d0), vmulq_f32(load_scale(p0), qs));
      f1 = vfmaq_f32(f1, vcvtq_f32_s32(d1), vmulq_f32(load_scale(p1), qs));
      f2 = vfmaq_f32(f2, vcvtq_f32_s32(d2), vmulq_f32(load_scale(p2), qs));
      f3 = vfmaq_f32(f3, vcvtq_f32_s32(d3), vmulq_f32(load_scale(p3), qs));
      if constexpr (HasZero) {
        const float32x4_t qz = vld1q_f32(q_bsum + b);
        const auto load_zero =
            [](const uint8_t* base) noexcept -> float32x4_t {
          if constexpr (ZeroF16) {
            return vcvt_f32_f16(
                vld1_f16(reinterpret_cast<const __fp16*>(base)));
          } else {
            return vld1q_f32(reinterpret_cast<const float*>(base));
          }
        };
        f0 = vfmaq_f32(f0, load_zero(p0 + s_step), qz);
        f1 = vfmaq_f32(f1, load_zero(p1 + s_step), qz);
        f2 = vfmaq_f32(f2, load_zero(p2 + s_step), qz);
        f3 = vfmaq_f32(f3, load_zero(p3 + s_step), qz);
      }
    }
    p0 += s_step + z_step;
    p1 += s_step + z_step;
    p2 += s_step + z_step;
    p3 += s_step + z_step;
  }
  out[0] = vaddvq_f32(f0);
  out[1] = vaddvq_f32(f1);
  out[2] = vaddvq_f32(f2);
  out[3] = vaddvq_f32(f3);
}

YNN_HNSW_TARGET_DOTPROD int32_t dot_i8_block_neon(const int8_t* q_ptr,
                                                  const int8_t* w_ptr,
                                                  size_t G) noexcept {
  int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
  int32x4_t acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
  size_t c = 0;
  for (; c + 64 <= G; c += 64) {
    acc0 = vdotq_s32(acc0, vld1q_s8(q_ptr + c), vld1q_s8(w_ptr + c));
    acc1 = vdotq_s32(acc1, vld1q_s8(q_ptr + c + 16), vld1q_s8(w_ptr + c + 16));
    acc2 = vdotq_s32(acc2, vld1q_s8(q_ptr + c + 32), vld1q_s8(w_ptr + c + 32));
    acc3 = vdotq_s32(acc3, vld1q_s8(q_ptr + c + 48), vld1q_s8(w_ptr + c + 48));
  }
  int32_t dot_i32 =
      vaddvq_s32(vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3)));
  for (; c < G; ++c) {
    dot_i32 += static_cast<int32_t>(q_ptr[c]) * static_cast<int32_t>(w_ptr[c]);
  }
  return dot_i32;
}

YNN_HNSW_TARGET_DOTPROD void dot4_i8_slices_neon(
    const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3,
    size_t B, size_t G, size_t row_bytes, bool scale_fp16, bool zero_fp16,
    bool has_zero, const int8_t* q_i8, const float* q_scale,
    const float* q_bsum, float out[4]) noexcept {
  const auto* w0 = reinterpret_cast<const int8_t*>(s0);
  const auto* w1 = reinterpret_cast<const int8_t*>(s1);
  const auto* w2 = reinterpret_cast<const int8_t*>(s2);
  const auto* w3 = reinterpret_cast<const int8_t*>(s3);
  const uint8_t* sc0 = s0 + row_bytes;
  const uint8_t* sc1 = s1 + row_bytes;
  const uint8_t* sc2 = s2 + row_bytes;
  const uint8_t* sc3 = s3 + row_bytes;
  const size_t wsum_off = row_bytes + B * (scale_fp16 ? 2u : 4u) +
                          (has_zero ? B * (zero_fp16 ? 2u : 4u) : 0u);
  out[0] = out[1] = out[2] = out[3] = 0.0f;
  for (size_t b = 0; b < B; ++b) {
    const int8_t* q_ptr = q_i8 + b * G;
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
    int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    size_t c = 0;
    for (; c + 32 <= G; c += 32) {
      const int8x16_t qv0 = vld1q_s8(q_ptr + c);
      const int8x16_t qv1 = vld1q_s8(q_ptr + c + 16);
      a0 = vdotq_s32(vdotq_s32(a0, qv0, vld1q_s8(w0 + b * G + c)), qv1,
                     vld1q_s8(w0 + b * G + c + 16));
      a1 = vdotq_s32(vdotq_s32(a1, qv0, vld1q_s8(w1 + b * G + c)), qv1,
                     vld1q_s8(w1 + b * G + c + 16));
      a2 = vdotq_s32(vdotq_s32(a2, qv0, vld1q_s8(w2 + b * G + c)), qv1,
                     vld1q_s8(w2 + b * G + c + 16));
      a3 = vdotq_s32(vdotq_s32(a3, qv0, vld1q_s8(w3 + b * G + c)), qv1,
                     vld1q_s8(w3 + b * G + c + 16));
    }
    int32_t d0 = vaddvq_s32(a0), d1 = vaddvq_s32(a1);
    int32_t d2 = vaddvq_s32(a2), d3 = vaddvq_s32(a3);
    for (; c < G; ++c) {
      const int32_t qv = static_cast<int32_t>(q_ptr[c]);
      d0 += qv * w0[b * G + c];
      d1 += qv * w1[b * G + c];
      d2 += qv * w2[b * G + c];
      d3 += qv * w3[b * G + c];
    }
    const int32_t zp = static_cast<int32_t>(q_bsum[b]);
    if (zp != 0) {
      int32_t ws0 = 0, ws1 = 0, ws2 = 0, ws3 = 0;
      std::memcpy(&ws0, s0 + wsum_off + b * sizeof(int32_t), sizeof(int32_t));
      std::memcpy(&ws1, s1 + wsum_off + b * sizeof(int32_t), sizeof(int32_t));
      std::memcpy(&ws2, s2 + wsum_off + b * sizeof(int32_t), sizeof(int32_t));
      std::memcpy(&ws3, s3 + wsum_off + b * sizeof(int32_t), sizeof(int32_t));
      d0 -= zp * ws0;
      d1 -= zp * ws1;
      d2 -= zp * ws2;
      d3 -= zp * ws3;
    }
    out[0] += static_cast<float>(d0) * (q_scale[b] * read_meta(sc0, scale_fp16, b));
    out[1] += static_cast<float>(d1) * (q_scale[b] * read_meta(sc1, scale_fp16, b));
    out[2] += static_cast<float>(d2) * (q_scale[b] * read_meta(sc2, scale_fp16, b));
    out[3] += static_cast<float>(d3) * (q_scale[b] * read_meta(sc3, scale_fp16, b));
  }
}

YNN_HNSW_TARGET_DOTPROD int32_t dot_i4_block_neon(const uint8_t* w_block,
                                                  const int8_t* q_block,
                                                  size_t chunks) noexcept {
  int32x4_t acc = vdupq_n_s32(0);
  for (size_t c = 0; c < chunks; ++c) {
    const uint8x16_t raw = vld1q_u8(w_block + c * 16);
    const int8x16_t we = vshrq_n_s8(vreinterpretq_s8_u8(vshlq_n_u8(raw, 4)), 4);
    const int8x16_t wo = vshrq_n_s8(vreinterpretq_s8_u8(raw), 4);
    const int8_t* qp = q_block + c * 32;
    acc = vdotq_s32(vdotq_s32(acc, we, vld1q_s8(qp)), wo, vld1q_s8(qp + 16));
  }
  return vaddvq_s32(acc);
}

YNN_HNSW_TARGET_DOTPROD int32_t dot_i2_block_neon(const uint8_t* wp,
                                                  const int8_t* qp,
                                                  size_t quarter) noexcept {
  int32x4_t acc = vdupq_n_s32(0);
  size_t c = 0;
  for (; c + 16 <= quarter; c += 16) {
    const int8x16_t raw = vreinterpretq_s8_u8(vld1q_u8(wp + c));
    const int8x16_t w0 = vshrq_n_s8(vshlq_n_s8(raw, 6), 6);
    const int8x16_t w1 = vshrq_n_s8(vshlq_n_s8(raw, 4), 6);
    const int8x16_t w2 = vshrq_n_s8(vshlq_n_s8(raw, 2), 6);
    const int8x16_t w3 = vshrq_n_s8(raw, 6);
    acc = vdotq_s32(acc, w0, vld1q_s8(qp + 0 * quarter + c));
    acc = vdotq_s32(acc, w1, vld1q_s8(qp + 1 * quarter + c));
    acc = vdotq_s32(acc, w2, vld1q_s8(qp + 2 * quarter + c));
    acc = vdotq_s32(acc, w3, vld1q_s8(qp + 3 * quarter + c));
  }
  int32_t dot_i32 = vaddvq_s32(acc);
  for (; c < quarter; ++c) {
    const auto raw = static_cast<int8_t>(wp[c]);
    dot_i32 += static_cast<int32_t>(static_cast<int8_t>(raw << 6) >> 6) *
                   qp[0 * quarter + c] +
               static_cast<int32_t>(static_cast<int8_t>(raw << 4) >> 6) *
                   qp[1 * quarter + c] +
               static_cast<int32_t>(static_cast<int8_t>(raw << 2) >> 6) *
                   qp[2 * quarter + c] +
               static_cast<int32_t>(raw >> 6) * qp[3 * quarter + c];
  }
  return dot_i32;
}

YNN_HNSW_TARGET_DOTPROD void dot4_i2_slices_neon(
    const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const uint8_t* s3,
    size_t B, size_t G, size_t row_bytes, bool scale_fp16, const int8_t* q_i8,
    const float* q_scale, float out[4]) noexcept {
  const size_t quarter = G / 4;
  const uint8_t* sc0 = s0 + row_bytes;
  const uint8_t* sc1 = s1 + row_bytes;
  const uint8_t* sc2 = s2 + row_bytes;
  const uint8_t* sc3 = s3 + row_bytes;
  out[0] = out[1] = out[2] = out[3] = 0.0f;
  for (size_t b = 0; b < B; ++b) {
    const int8_t* qp = q_i8 + b * G;
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
    int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    size_t c = 0;
    for (; c + 16 <= quarter; c += 16) {
      const int8x16_t qv0 = vld1q_s8(qp + 0 * quarter + c);
      const int8x16_t qv1 = vld1q_s8(qp + 1 * quarter + c);
      const int8x16_t qv2 = vld1q_s8(qp + 2 * quarter + c);
      const int8x16_t qv3 = vld1q_s8(qp + 3 * quarter + c);
      const auto dot_row = [&](int32x4_t acc, const uint8_t* wp) noexcept {
        const int8x16_t raw = vreinterpretq_s8_u8(vld1q_u8(wp + b * quarter + c));
        acc = vdotq_s32(acc, vshrq_n_s8(vshlq_n_s8(raw, 6), 6), qv0);
        acc = vdotq_s32(acc, vshrq_n_s8(vshlq_n_s8(raw, 4), 6), qv1);
        acc = vdotq_s32(acc, vshrq_n_s8(vshlq_n_s8(raw, 2), 6), qv2);
        return vdotq_s32(acc, vshrq_n_s8(raw, 6), qv3);
      };
      a0 = dot_row(a0, s0);
      a1 = dot_row(a1, s1);
      a2 = dot_row(a2, s2);
      a3 = dot_row(a3, s3);
    }
    int32_t d0 = vaddvq_s32(a0), d1 = vaddvq_s32(a1);
    int32_t d2 = vaddvq_s32(a2), d3 = vaddvq_s32(a3);
    for (; c < quarter; ++c) {
      const auto dot_tail = [&](const uint8_t* wp) noexcept {
        const auto raw = static_cast<int8_t>(wp[b * quarter + c]);
        return static_cast<int32_t>(static_cast<int8_t>(raw << 6) >> 6) *
                   qp[0 * quarter + c] +
               static_cast<int32_t>(static_cast<int8_t>(raw << 4) >> 6) *
                   qp[1 * quarter + c] +
               static_cast<int32_t>(static_cast<int8_t>(raw << 2) >> 6) *
                   qp[2 * quarter + c] +
               static_cast<int32_t>(raw >> 6) * qp[3 * quarter + c];
      };
      d0 += dot_tail(s0);
      d1 += dot_tail(s1);
      d2 += dot_tail(s2);
      d3 += dot_tail(s3);
    }
    out[0] += static_cast<float>(d0) * (q_scale[b] * read_meta(sc0, scale_fp16, b));
    out[1] += static_cast<float>(d1) * (q_scale[b] * read_meta(sc1, scale_fp16, b));
    out[2] += static_cast<float>(d2) * (q_scale[b] * read_meta(sc2, scale_fp16, b));
    out[3] += static_cast<float>(d3) * (q_scale[b] * read_meta(sc3, scale_fp16, b));
  }
}
#endif

}  // namespace

hnsw_dot_kernels get_hnsw_dot_kernels(bool bf16) noexcept {
  hnsw_dot_kernels kernels;
  kernels.dot = bf16 ? dot_portable<true> : dot_portable<false>;
  kernels.dot4 = bf16 ? dot4_portable<true> : dot4_portable<false>;
  kernels.row_dot = bf16 ? dot_rows_portable<true> : dot_rows_portable<false>;
  kernels.row_dot4 =
      bf16 ? dot_rows4_portable<true> : dot_rows4_portable<false>;
#if defined(YNN_HNSW_AVX512) && defined(YNN_ARCH_X86)
  if (is_arch_supported(arch_flag::avx512f | arch_flag::avx512bw |
                        arch_flag::fma3)) {
    kernels.dot = bf16 ? dot_avx512<true> : dot_avx512<false>;
    kernels.dot4 = bf16 ? dot4_avx512<true> : dot4_avx512<false>;
    kernels.row_dot = bf16 ? dot_rows_avx512<true> : dot_rows_avx512<false>;
    kernels.row_dot4 =
        bf16 ? dot_rows4_avx512<true> : dot_rows4_avx512<false>;
  }
#elif defined(YNN_HNSW_NEON)
  kernels.dot = bf16 ? dot_neon<true> : dot_neon<false>;
  kernels.dot4 = bf16 ? dot4_neon<true> : dot4_neon<false>;
  if (bf16 && has_arm64_bf16()) {
#if defined(YNN_HNSW_ARM64_TARGETS)
    kernels.row_dot = dot_rows_bf16_neon;
    kernels.row_dot4 = dot_rows4_bf16_neon;
#endif
  } else {
    kernels.row_dot = bf16 ? dot_rows_neon<true> : dot_rows_neon<false>;
    kernels.row_dot4 = bf16 ? dot_rows4_neon<true> : dot_rows4_neon<false>;
  }
#endif
  return kernels;
}

float hnsw_weight_element_f32(const void* row, size_t column,
                              bool bf16) noexcept {
  return bf16 ? weight_elem<true>(row, column)
              : weight_elem<false>(row, column);
}

void hnsw_prefetch_slice(const void* slice_ptr, size_t slice_bytes) noexcept {
  const size_t limit = std::min<size_t>(slice_bytes, 1024);
#if defined(__GNUC__) || defined(__clang__)
  const auto* base = static_cast<const char*>(slice_ptr);
  for (size_t off = 0; off < limit; off += 128) {
    __builtin_prefetch(base + off, 0, 1);
  }
#else
  (void)slice_ptr;
  (void)limit;
#endif
}

size_t hnsw_quantized_slice_bytes(size_t width, int32_t bits,
                                  int32_t group_size, bool scale_fp16,
                                  bool zero_fp16, bool has_zero) noexcept {
  const size_t row_bytes = width / (8u / static_cast<size_t>(bits));
  const size_t B = width / static_cast<size_t>(group_size);
  const size_t scale_row_bytes = B * (scale_fp16 ? 2u : 4u);
  const size_t zero_row_bytes = has_zero ? B * (zero_fp16 ? 2u : 4u) : 0u;
  const size_t wsum_row_bytes = bits == 8 ? B * sizeof(int32_t) : 0u;
  const size_t raw =
      row_bytes + scale_row_bytes + zero_row_bytes + wsum_row_bytes;
  return raw >= 64 ? ((raw + 63u) & ~size_t{63}) : ((raw + 15u) & ~size_t{15});
}

void hnsw_pack_quantized_slice(uint8_t* dst, const void* w_row,
                               const void* s_row, const void* z_row,
                               size_t width, int32_t bits, int32_t group_size,
                               bool scale_fp16, bool zero_fp16,
                               bool has_zero) noexcept {
  const size_t row_bytes = width / (8u / static_cast<size_t>(bits));
  const size_t G = static_cast<size_t>(group_size);
  const size_t B = width / G;
  const size_t s_elem = scale_fp16 ? 2u : 4u;
  const size_t z_elem = zero_fp16 ? 2u : 4u;
  const auto* w_bytes = static_cast<const uint8_t*>(w_row);
  const auto* s_bytes = static_cast<const uint8_t*>(s_row);
  const auto* z_bytes = static_cast<const uint8_t*>(z_row);
  if (bits == 4 && B % 4 == 0 && has_arm64_dotprod_fp16()) {
    const size_t chunks = G / 32;
    const size_t words_per_block = chunks * 4;
    uint8_t* ptr = dst;
    for (size_t b = 0; b < B; b += 4) {
      for (size_t k = 0; k < words_per_block; ++k) {
        for (size_t r = 0; r < 4; ++r) {
          std::memcpy(ptr + (k * 4 + r) * 4,
                      w_bytes + ((b + r) * words_per_block + k) * 4, 4);
        }
      }
      ptr += words_per_block * 16;
      std::memcpy(ptr, s_bytes + b * s_elem, 4 * s_elem);
      ptr += 4 * s_elem;
      if (has_zero) {
        std::memcpy(ptr, z_bytes + b * z_elem, 4 * z_elem);
        ptr += 4 * z_elem;
      }
    }
    return;
  }
  const size_t scale_row_bytes = B * s_elem;
  const size_t zero_row_bytes = has_zero ? B * z_elem : 0u;
  std::memcpy(dst, w_bytes, row_bytes);
  std::memcpy(dst + row_bytes, s_bytes, scale_row_bytes);
  if (has_zero) {
    std::memcpy(dst + row_bytes + scale_row_bytes, z_bytes, zero_row_bytes);
  }
  if (bits == 8) {
    const auto* w_i8 = static_cast<const int8_t*>(w_row);
    uint8_t* wsum_dst = dst + row_bytes + scale_row_bytes + zero_row_bytes;
    for (size_t b = 0; b < B; ++b) {
      int32_t wsum = 0;
      for (size_t c = 0; c < G; ++c) {
        wsum += static_cast<int32_t>(w_i8[b * G + c]);
      }
      std::memcpy(wsum_dst + b * sizeof(int32_t), &wsum, sizeof(int32_t));
    }
  }
}

void hnsw_prepare_quantized_query(const float* query, size_t width,
                                  int32_t bits, int32_t group_size,
                                  float input_scale, std::vector<int8_t>& q_i8,
                                  std::vector<float>& q_scale,
                                  std::vector<float>& q_bsum) {
  if (bits == 0) return;
  const size_t G = static_cast<size_t>(group_size);
  const size_t B = width / G;
  const bool interleave_i4 =
      bits == 4 && B % 4 == 0 && has_arm64_dotprod_fp16();
  q_i8.resize(width);
  q_scale.resize(B);
  q_bsum.resize(B);
  for (size_t b = 0; b < B; ++b) {
    const float* x = query + b * G;
    if (bits == 8 && input_scale <= 0.0f) {
      float rmin = 0.0f, rmax = 0.0f;
      for (size_t i = 0; i < G; ++i) {
        rmin = std::min(rmin, x[i]);
        rmax = std::max(rmax, x[i]);
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
        const float fzp = min_error + max_error > 0.0f
                              ? -128.0f - rmin * inv_scale
                              : 127.0f - rmax * inv_scale;
        zp = static_cast<int32_t>(std::rint(std::clamp(fzp, -128.0f, 127.0f)));
      }
      q_scale[b] = scale;
      q_bsum[b] = static_cast<float>(zp);
      int8_t* dst = q_i8.data() + b * G;
      for (size_t i = 0; i < G; ++i) {
        const int32_t q =
            static_cast<int32_t>(std::round(x[i] * inv_scale)) + zp;
        dst[i] = static_cast<int8_t>(std::clamp<int32_t>(q, -128, 127));
      }
      continue;
    }
    float scale = input_scale;
    float inv_scale = 0.0f;
    if (scale > 0.0f) {
      inv_scale = 1.0f / scale;
    } else {
      float max_abs = 0.0f;
      for (size_t i = 0; i < G; ++i) max_abs = std::max(max_abs, std::abs(x[i]));
      scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
      inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 0.0f;
    }
    q_scale[b] = scale;
    const float min_q = input_scale > 0.0f ? -128.0f : -127.0f;
    if (bits == 8) {
      int8_t* dst = q_i8.data() + b * G;
      for (size_t i = 0; i < G; ++i) {
        dst[i] = static_cast<int8_t>(
            std::clamp(std::nearbyint(x[i] * inv_scale), min_q, 127.0f));
      }
    } else if (bits == 4) {
      const size_t chunks = G / 32;
      int32_t bsum = 0;
      if (interleave_i4) {
        const size_t words_per_block = chunks * 4;
        const size_t base_k = (b / 4) * words_per_block;
        const size_t r = b & 3u;
        for (size_t c = 0; c < chunks; ++c) {
          const float* xc = x + c * 32;
          for (size_t i = 0; i < 16; ++i) {
            const auto qe = static_cast<int8_t>(std::clamp(
                std::nearbyint(xc[2 * i] * inv_scale), min_q, 127.0f));
            const auto qo = static_cast<int8_t>(std::clamp(
                std::nearbyint(xc[2 * i + 1] * inv_scale), min_q, 127.0f));
            const size_t k = c * 4 + (i >> 2u);
            const size_t lane = i & 3u;
            int8_t* dst = q_i8.data() + (base_k + k) * 32 + r * 4 + lane;
            dst[0] = qe;
            dst[16] = qo;
            bsum += static_cast<int32_t>(qe) + static_cast<int32_t>(qo);
          }
        }
        q_bsum[b] = static_cast<float>(bsum) * scale;
        continue;
      }
      for (size_t c = 0; c < chunks; ++c) {
        const float* xc = x + c * 32;
        int8_t* dst = q_i8.data() + (b * chunks + c) * 32;
        for (size_t i = 0; i < 16; ++i) {
          const auto qe = static_cast<int8_t>(
              std::clamp(std::nearbyint(xc[2 * i] * inv_scale), min_q, 127.0f));
          const auto qo = static_cast<int8_t>(std::clamp(
              std::nearbyint(xc[2 * i + 1] * inv_scale), min_q, 127.0f));
          dst[i] = qe;
          dst[16 + i] = qo;
          bsum += static_cast<int32_t>(qe) + static_cast<int32_t>(qo);
        }
      }
      q_bsum[b] = static_cast<float>(bsum) * scale;
    } else if (bits == 2) {
      const size_t quarter = G / 4;
      int8_t* dst = q_i8.data() + b * G;
      for (size_t i = 0; i < quarter; ++i) {
        dst[0 * quarter + i] = static_cast<int8_t>(
            std::clamp(std::nearbyint(x[4 * i + 0] * inv_scale), min_q, 127.0f));
        dst[1 * quarter + i] = static_cast<int8_t>(
            std::clamp(std::nearbyint(x[4 * i + 1] * inv_scale), min_q, 127.0f));
        dst[2 * quarter + i] = static_cast<int8_t>(
            std::clamp(std::nearbyint(x[4 * i + 2] * inv_scale), min_q, 127.0f));
        dst[3 * quarter + i] = static_cast<int8_t>(
            std::clamp(std::nearbyint(x[4 * i + 3] * inv_scale), min_q, 127.0f));
      }
    }
  }
}

float hnsw_quantized_slice_dot(const uint8_t* slice_ptr, size_t width,
                               int32_t bits, int32_t group_size,
                               bool scale_fp16, bool zero_fp16, bool has_zero,
                               const int8_t* q_i8, const float* q_scale,
                               const float* q_bsum) noexcept {
  const size_t G = static_cast<size_t>(group_size);
  const size_t B = width / G;
  const size_t row_bytes = width / (8u / static_cast<size_t>(bits));
  const size_t scale_row_bytes = B * (scale_fp16 ? 2u : 4u);
  const uint8_t* scale_ptr = slice_ptr + row_bytes;
  const uint8_t* zero_ptr = scale_ptr + scale_row_bytes;
  float score = 0.0f;
  if (bits == 8) {
    const size_t zero_row_bytes = has_zero ? B * (zero_fp16 ? 2u : 4u) : 0u;
    const uint8_t* wsum_ptr = zero_ptr + zero_row_bytes;
    const bool use_dotprod = has_arm64_dotprod();
    for (size_t b = 0; b < B; ++b) {
      const int8_t* q_ptr = q_i8 + b * G;
      const auto* w_ptr = reinterpret_cast<const int8_t*>(slice_ptr + b * G);
      int32_t dot_i32 = 0;
#if defined(YNN_HNSW_ARM64_TARGETS)
      if (use_dotprod) {
        dot_i32 = dot_i8_block_neon(q_ptr, w_ptr, G);
      } else
#endif
      {
        (void)use_dotprod;
        for (size_t c = 0; c < G; ++c) {
          dot_i32 += static_cast<int32_t>(q_ptr[c]) *
                     static_cast<int32_t>(w_ptr[c]);
        }
      }
      const int32_t zp = static_cast<int32_t>(q_bsum[b]);
      if (zp != 0) {
        int32_t wsum = 0;
        std::memcpy(&wsum, wsum_ptr + b * sizeof(int32_t), sizeof(int32_t));
        dot_i32 -= zp * wsum;
      }
      score += static_cast<float>(dot_i32) *
               (q_scale[b] * read_meta(scale_ptr, scale_fp16, b));
    }
  } else if (bits == 4) {
    const size_t chunks = G / 32;
#if defined(YNN_HNSW_ARM64_TARGETS)
    if (B % 4 == 0 && has_arm64_dotprod_fp16()) {
      if (!has_zero) {
        return scale_fp16 ? quantized_slice_dot_int4_neon<true, false, false>(
                                slice_ptr, B, chunks, q_i8, q_scale, q_bsum)
                          : quantized_slice_dot_int4_neon<false, false, false>(
                                slice_ptr, B, chunks, q_i8, q_scale, q_bsum);
      }
      if (scale_fp16) {
        return zero_fp16 ? quantized_slice_dot_int4_neon<true, true, true>(
                               slice_ptr, B, chunks, q_i8, q_scale, q_bsum)
                         : quantized_slice_dot_int4_neon<true, true, false>(
                               slice_ptr, B, chunks, q_i8, q_scale, q_bsum);
      }
      return zero_fp16 ? quantized_slice_dot_int4_neon<false, true, true>(
                             slice_ptr, B, chunks, q_i8, q_scale, q_bsum)
                       : quantized_slice_dot_int4_neon<false, true, false>(
                             slice_ptr, B, chunks, q_i8, q_scale, q_bsum);
    }
#endif
    const bool use_dotprod = has_arm64_dotprod();
    for (size_t b = 0; b < B; ++b) {
      int32_t dot_i32 = 0;
#if defined(YNN_HNSW_ARM64_TARGETS)
      if (use_dotprod) {
        dot_i32 = dot_i4_block_neon(slice_ptr + b * chunks * 16,
                                    q_i8 + b * chunks * 32, chunks);
      } else
#endif
      {
        (void)use_dotprod;
        for (size_t c = 0; c < chunks; ++c) {
          const uint8_t* wp = slice_ptr + (b * chunks + c) * 16;
          const int8_t* qp = q_i8 + (b * chunks + c) * 32;
          for (size_t i = 0; i < 16; ++i) {
            const uint8_t byte = wp[i];
            const auto we =
                static_cast<int8_t>(static_cast<uint8_t>(byte << 4)) >> 4;
            const auto wo = static_cast<int8_t>(byte) >> 4;
            dot_i32 +=
                static_cast<int32_t>(we) * qp[i] +
                static_cast<int32_t>(wo) * qp[16 + i];
          }
        }
      }
      score += static_cast<float>(dot_i32) *
               (q_scale[b] * read_meta(scale_ptr, scale_fp16, b));
      if (has_zero) {
        score += q_bsum[b] * read_meta(zero_ptr, zero_fp16, b);
      }
    }
  } else if (bits == 2) {
    const size_t quarter = G / 4;
    const bool use_dotprod = has_arm64_dotprod();
    for (size_t b = 0; b < B; ++b) {
      const uint8_t* wp = slice_ptr + b * quarter;
      const int8_t* qp = q_i8 + b * G;
      int32_t dot_i32 = 0;
#if defined(YNN_HNSW_ARM64_TARGETS)
      if (use_dotprod) {
        dot_i32 = dot_i2_block_neon(wp, qp, quarter);
      } else
#endif
      {
        (void)use_dotprod;
        for (size_t c = 0; c < quarter; ++c) {
          const auto raw = static_cast<int8_t>(wp[c]);
          dot_i32 += static_cast<int32_t>(static_cast<int8_t>(raw << 6) >> 6) *
                         qp[0 * quarter + c] +
                     static_cast<int32_t>(static_cast<int8_t>(raw << 4) >> 6) *
                         qp[1 * quarter + c] +
                     static_cast<int32_t>(static_cast<int8_t>(raw << 2) >> 6) *
                         qp[2 * quarter + c] +
                     static_cast<int32_t>(raw >> 6) * qp[3 * quarter + c];
        }
      }
      score += static_cast<float>(dot_i32) *
               (q_scale[b] * read_meta(scale_ptr, scale_fp16, b));
    }
  }
  return score;
}

void hnsw_quantized_slices4_dot(const uint8_t* s0, const uint8_t* s1,
                                const uint8_t* s2, const uint8_t* s3,
                                size_t width, int32_t bits, int32_t group_size,
                                bool scale_fp16, bool zero_fp16, bool has_zero,
                                const int8_t* q_i8, const float* q_scale,
                                const float* q_bsum, float out[4]) noexcept {
#if defined(YNN_HNSW_ARM64_TARGETS)
  const size_t G = static_cast<size_t>(group_size);
  const size_t B = width / G;
  const size_t row_bytes = width / (8u / static_cast<size_t>(bits));
  if (bits == 4 && B % 4 == 0 && has_arm64_dotprod_fp16()) {
    const size_t chunks = G / 32;
    if (!has_zero) {
      if (scale_fp16) {
        quantized_slices4_dot_int4_neon<true, false, false>(
            s0, s1, s2, s3, B, chunks, q_i8, q_scale, q_bsum, out);
      } else {
        quantized_slices4_dot_int4_neon<false, false, false>(
            s0, s1, s2, s3, B, chunks, q_i8, q_scale, q_bsum, out);
      }
      return;
    }
    if (scale_fp16) {
      if (zero_fp16) {
        quantized_slices4_dot_int4_neon<true, true, true>(
            s0, s1, s2, s3, B, chunks, q_i8, q_scale, q_bsum, out);
      } else {
        quantized_slices4_dot_int4_neon<true, true, false>(
            s0, s1, s2, s3, B, chunks, q_i8, q_scale, q_bsum, out);
      }
      return;
    }
    if (zero_fp16) {
      quantized_slices4_dot_int4_neon<false, true, true>(
          s0, s1, s2, s3, B, chunks, q_i8, q_scale, q_bsum, out);
    } else {
      quantized_slices4_dot_int4_neon<false, true, false>(
          s0, s1, s2, s3, B, chunks, q_i8, q_scale, q_bsum, out);
    }
    return;
  }
  if (bits == 8 && has_arm64_dotprod()) {
    dot4_i8_slices_neon(s0, s1, s2, s3, B, G, row_bytes, scale_fp16, zero_fp16,
                        has_zero, q_i8, q_scale, q_bsum, out);
    return;
  }
  if (bits == 2 && has_arm64_dotprod()) {
    dot4_i2_slices_neon(s0, s1, s2, s3, B, G, row_bytes, scale_fp16, q_i8,
                        q_scale, out);
    return;
  }
#endif
  out[0] = hnsw_quantized_slice_dot(s0, width, bits, group_size, scale_fp16,
                                    zero_fp16, has_zero, q_i8, q_scale, q_bsum);
  out[1] = hnsw_quantized_slice_dot(s1, width, bits, group_size, scale_fp16,
                                    zero_fp16, has_zero, q_i8, q_scale, q_bsum);
  out[2] = hnsw_quantized_slice_dot(s2, width, bits, group_size, scale_fp16,
                                    zero_fp16, has_zero, q_i8, q_scale, q_bsum);
  out[3] = hnsw_quantized_slice_dot(s3, width, bits, group_size, scale_fp16,
                                    zero_fp16, has_zero, q_i8, q_scale, q_bsum);
}

}  // namespace ynn
