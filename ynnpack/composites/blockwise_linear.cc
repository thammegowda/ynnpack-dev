// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ynnpack/base/arch.h"
#include "ynnpack/composites/composites.h"

#if defined(__ARM_NEON) && (defined(__aarch64__) || defined(__arm64__))
#include <arm_neon.h>
#define YNN_BLOCKWISE_ARM64 1
#endif

namespace ynn {
namespace {

constexpr size_t NR_TILE = 32;

#if defined(YNN_BLOCKWISE_ARM64)
__attribute__((target("dotprod,fp16"))) void store_row32_f16(
    float* dst, const float32x4_t acc[8], size_t n_valid,
    float output_scale) noexcept {
  for (size_t j = 0; j < 8; ++j) {
    if (j * 4 >= n_valid) break;
    float32x4_t res = acc[j];
    if (output_scale > 0.0f) {
      const float32x4_t vinv = vdupq_n_f32(1.0f / output_scale);
      const float32x4_t vscale = vdupq_n_f32(output_scale);
      res = vmulq_f32(vminq_f32(vmaxq_f32(vrndnq_f32(vmulq_f32(res, vinv)),
                                          vdupq_n_f32(-128.0f)),
                                vdupq_n_f32(127.0f)),
                      vscale);
    }
    if (j * 4 + 4 <= n_valid) {
      vst1q_f32(dst + j * 4, res);
    } else {
      alignas(16) float tmp[4];
      vst1q_f32(tmp, res);
      for (size_t i = 0; j * 4 + i < n_valid; ++i) dst[j * 4 + i] = tmp[i];
    }
  }
}

__attribute__((target("dotprod"))) void store_row32_i8(
    float* dst, const float32x4_t acc[8], size_t n_valid,
    float output_scale) noexcept {
  for (size_t j = 0; j < 8; ++j) {
    if (j * 4 >= n_valid) break;
    float32x4_t res = acc[j];
    if (output_scale > 0.0f) {
      const float32x4_t vinv = vdupq_n_f32(1.0f / output_scale);
      const float32x4_t vscale = vdupq_n_f32(output_scale);
      res = vmulq_f32(vminq_f32(vmaxq_f32(vrndnq_f32(vmulq_f32(res, vinv)),
                                          vdupq_n_f32(-128.0f)),
                                vdupq_n_f32(127.0f)),
                      vscale);
    }
    if (j * 4 + 4 <= n_valid) {
      vst1q_f32(dst + j * 4, res);
    } else {
      alignas(16) float tmp[4];
      vst1q_f32(tmp, res);
      for (size_t i = 0; j * 4 + i < n_valid; ++i) dst[j * 4 + i] = tmp[i];
    }
  }
}

#define YNN_I4_DOT32_STEP(STEP, P, QE, QO, D)                                  \
  do {                                                                         \
    const int8x16_t r0 = vld1q_s8(reinterpret_cast<const int8_t*>((P) + 0));   \
    const int8x16_t r1 = vld1q_s8(reinterpret_cast<const int8_t*>((P) + 16));  \
    const int8x16_t r2 = vld1q_s8(reinterpret_cast<const int8_t*>((P) + 32));  \
    const int8x16_t r3 = vld1q_s8(reinterpret_cast<const int8_t*>((P) + 48));  \
    (D)[0] =                                                                   \
        vdotq_laneq_s32((D)[0], vshrq_n_s8(vshlq_n_s8(r0, 4), 4), (QE), STEP); \
    (D)[0] = vdotq_laneq_s32((D)[0], vshrq_n_s8(r0, 4), (QO), STEP);           \
    (D)[1] =                                                                   \
        vdotq_laneq_s32((D)[1], vshrq_n_s8(vshlq_n_s8(r1, 4), 4), (QE), STEP); \
    (D)[1] = vdotq_laneq_s32((D)[1], vshrq_n_s8(r1, 4), (QO), STEP);           \
    (D)[2] =                                                                   \
        vdotq_laneq_s32((D)[2], vshrq_n_s8(vshlq_n_s8(r2, 4), 4), (QE), STEP); \
    (D)[2] = vdotq_laneq_s32((D)[2], vshrq_n_s8(r2, 4), (QO), STEP);           \
    (D)[3] =                                                                   \
        vdotq_laneq_s32((D)[3], vshrq_n_s8(vshlq_n_s8(r3, 4), 4), (QE), STEP); \
    (D)[3] = vdotq_laneq_s32((D)[3], vshrq_n_s8(r3, 4), (QO), STEP);           \
    const int8x16_t r4 = vld1q_s8(reinterpret_cast<const int8_t*>((P) + 64));  \
    const int8x16_t r5 = vld1q_s8(reinterpret_cast<const int8_t*>((P) + 80));  \
    const int8x16_t r6 = vld1q_s8(reinterpret_cast<const int8_t*>((P) + 96));  \
    const int8x16_t r7 = vld1q_s8(reinterpret_cast<const int8_t*>((P) + 112)); \
    (P) += 128;                                                                \
    (D)[4] =                                                                   \
        vdotq_laneq_s32((D)[4], vshrq_n_s8(vshlq_n_s8(r4, 4), 4), (QE), STEP); \
    (D)[4] = vdotq_laneq_s32((D)[4], vshrq_n_s8(r4, 4), (QO), STEP);           \
    (D)[5] =                                                                   \
        vdotq_laneq_s32((D)[5], vshrq_n_s8(vshlq_n_s8(r5, 4), 4), (QE), STEP); \
    (D)[5] = vdotq_laneq_s32((D)[5], vshrq_n_s8(r5, 4), (QO), STEP);           \
    (D)[6] =                                                                   \
        vdotq_laneq_s32((D)[6], vshrq_n_s8(vshlq_n_s8(r6, 4), 4), (QE), STEP); \
    (D)[6] = vdotq_laneq_s32((D)[6], vshrq_n_s8(r6, 4), (QO), STEP);           \
    (D)[7] =                                                                   \
        vdotq_laneq_s32((D)[7], vshrq_n_s8(vshlq_n_s8(r7, 4), 4), (QE), STEP); \
    (D)[7] = vdotq_laneq_s32((D)[7], vshrq_n_s8(r7, 4), (QO), STEP);           \
  } while (0)

#define YNN_I4_DOT32x2_STEP(STEP, P, QE0, QO0, QE1, QO1, D0, D1)               \
  do {                                                                         \
    for (size_t g = 0; g < 8; g += 2) {                                        \
      const int8x16_t ra =                                                     \
          vld1q_s8(reinterpret_cast<const int8_t*>((P) + g * 16));             \
      const int8x16_t rb =                                                     \
          vld1q_s8(reinterpret_cast<const int8_t*>((P) + g * 16 + 16));        \
      const int8x16_t wa_e = vshrq_n_s8(vshlq_n_s8(ra, 4), 4),                 \
                      wa_o = vshrq_n_s8(ra, 4);                                \
      const int8x16_t wb_e = vshrq_n_s8(vshlq_n_s8(rb, 4), 4),                 \
                      wb_o = vshrq_n_s8(rb, 4);                                \
      (D0)[g] = vdotq_laneq_s32(vdotq_laneq_s32((D0)[g], wa_e, (QE0), STEP),   \
                                wa_o, (QO0), STEP);                            \
      (D1)[g] = vdotq_laneq_s32(vdotq_laneq_s32((D1)[g], wa_e, (QE1), STEP),   \
                                wa_o, (QO1), STEP);                            \
      (D0)[g + 1] = vdotq_laneq_s32(                                           \
          vdotq_laneq_s32((D0)[g + 1], wb_e, (QE0), STEP), wb_o, (QO0), STEP); \
      (D1)[g + 1] = vdotq_laneq_s32(                                           \
          vdotq_laneq_s32((D1)[g + 1], wb_e, (QE1), STEP), wb_o, (QO1), STEP); \
    }                                                                          \
    (P) += 128;                                                                \
  } while (0)

#define YNN_I4_DOT16x4_STEP(STEP, P, QE, QO, D)                              \
  do {                                                                       \
    for (size_t g = 0; g < 4; ++g) {                                         \
      const int8x16_t raw =                                                  \
          vld1q_s8(reinterpret_cast<const int8_t*>((P) + g * 16));           \
      const int8x16_t we = vshrq_n_s8(vshlq_n_s8(raw, 4), 4),                \
                      wo = vshrq_n_s8(raw, 4);                               \
      (D)[0][g] = vdotq_laneq_s32(                                           \
          vdotq_laneq_s32((D)[0][g], we, (QE)[0], STEP), wo, (QO)[0], STEP); \
      (D)[1][g] = vdotq_laneq_s32(                                           \
          vdotq_laneq_s32((D)[1][g], we, (QE)[1], STEP), wo, (QO)[1], STEP); \
      (D)[2][g] = vdotq_laneq_s32(                                           \
          vdotq_laneq_s32((D)[2][g], we, (QE)[2], STEP), wo, (QO)[2], STEP); \
      (D)[3][g] = vdotq_laneq_s32(                                           \
          vdotq_laneq_s32((D)[3][g], we, (QE)[3], STEP), wo, (QO)[3], STEP); \
    }                                                                        \
  } while (0)

template <size_t CHUNKS, bool HAS_ZERO>
__attribute__((target("dotprod,fp16"))) void eval_i4_tile_32x1(
    size_t B, size_t runtime_chunks, const uint8_t* p, const int8_t* qp,
    const float* a_scale_base, const float* a_bsum_base, float* y_base,
    size_t n_valid, float output_scale) noexcept {
  const size_t chunks = CHUNKS ? CHUNKS : runtime_chunks;
  float32x4_t acc[8];
  for (auto& v : acc) v = vdupq_n_f32(0.0f);

  for (size_t b = 0; b < B; ++b) {
    int32x4_t d[8];
    for (auto& v : d) v = vdupq_n_s32(0);
    for (size_t c = 0; c < chunks; ++c, qp += 32) {
      const int8x16_t qe = vld1q_s8(qp + 0);
      const int8x16_t qo = vld1q_s8(qp + 16);
      YNN_I4_DOT32_STEP(0, p, qe, qo, d);
      YNN_I4_DOT32_STEP(1, p, qe, qo, d);
      YNN_I4_DOT32_STEP(2, p, qe, qo, d);
      YNN_I4_DOT32_STEP(3, p, qe, qo, d);
    }
    const auto* sp = reinterpret_cast<const __fp16*>(p);
    const float16x8_t s0 = vld1q_f16(sp + 0), s1 = vld1q_f16(sp + 8);
    const float16x8_t s2 = vld1q_f16(sp + 16), s3 = vld1q_f16(sp + 24);
    p += 64;
    const float a_s = a_scale_base[b];
    acc[0] = vfmaq_f32(acc[0], vcvtq_f32_s32(d[0]),
                       vmulq_n_f32(vcvt_f32_f16(vget_low_f16(s0)), a_s));
    acc[1] = vfmaq_f32(acc[1], vcvtq_f32_s32(d[1]),
                       vmulq_n_f32(vcvt_high_f32_f16(s0), a_s));
    acc[2] = vfmaq_f32(acc[2], vcvtq_f32_s32(d[2]),
                       vmulq_n_f32(vcvt_f32_f16(vget_low_f16(s1)), a_s));
    acc[3] = vfmaq_f32(acc[3], vcvtq_f32_s32(d[3]),
                       vmulq_n_f32(vcvt_high_f32_f16(s1), a_s));
    acc[4] = vfmaq_f32(acc[4], vcvtq_f32_s32(d[4]),
                       vmulq_n_f32(vcvt_f32_f16(vget_low_f16(s2)), a_s));
    acc[5] = vfmaq_f32(acc[5], vcvtq_f32_s32(d[5]),
                       vmulq_n_f32(vcvt_high_f32_f16(s2), a_s));
    acc[6] = vfmaq_f32(acc[6], vcvtq_f32_s32(d[6]),
                       vmulq_n_f32(vcvt_f32_f16(vget_low_f16(s3)), a_s));
    acc[7] = vfmaq_f32(acc[7], vcvtq_f32_s32(d[7]),
                       vmulq_n_f32(vcvt_high_f32_f16(s3), a_s));
    if constexpr (HAS_ZERO) {
      const auto* zp = reinterpret_cast<const __fp16*>(p);
      const float16x8_t z0 = vld1q_f16(zp + 0), z1 = vld1q_f16(zp + 8);
      const float16x8_t z2 = vld1q_f16(zp + 16), z3 = vld1q_f16(zp + 24);
      p += 64;
      const float a_z = a_bsum_base[b];
      acc[0] = vfmaq_n_f32(acc[0], vcvt_f32_f16(vget_low_f16(z0)), a_z);
      acc[1] = vfmaq_n_f32(acc[1], vcvt_high_f32_f16(z0), a_z);
      acc[2] = vfmaq_n_f32(acc[2], vcvt_f32_f16(vget_low_f16(z1)), a_z);
      acc[3] = vfmaq_n_f32(acc[3], vcvt_high_f32_f16(z1), a_z);
      acc[4] = vfmaq_n_f32(acc[4], vcvt_f32_f16(vget_low_f16(z2)), a_z);
      acc[5] = vfmaq_n_f32(acc[5], vcvt_high_f32_f16(z2), a_z);
      acc[6] = vfmaq_n_f32(acc[6], vcvt_f32_f16(vget_low_f16(z3)), a_z);
      acc[7] = vfmaq_n_f32(acc[7], vcvt_high_f32_f16(z3), a_z);
    }
  }
  store_row32_f16(y_base, acc, n_valid, output_scale);
}

template <size_t CHUNKS, bool HAS_ZERO>
__attribute__((target("dotprod,fp16"))) void dispatch_i4_tiles(
    size_t M, size_t N, size_t B, size_t block_size, float output_scale,
    const uint8_t* packed_ptr, const int8_t* act_q, const float* act_scale,
    const float* act_bsum, float* y, size_t t_begin, size_t t_end) noexcept {
  const size_t chunks = CHUNKS ? CHUNKS : (block_size / 32);
  const size_t block_bytes = chunks * 512 + 64 + (HAS_ZERO ? 64 : 0);
  const size_t tile_bytes = B * block_bytes;
  const size_t G = chunks * 32;

  for (size_t tile = t_begin; tile < t_end; ++tile) {
    const size_t n0 = tile * 32;
    const size_t n_valid = std::min<size_t>(32, N - n0);
    const uint8_t* tile_ptr = packed_ptr + tile * tile_bytes;

    if (M == 1) {
      eval_i4_tile_32x1<CHUNKS, HAS_ZERO>(B, chunks, tile_ptr, act_q, act_scale,
                                          act_bsum, y + n0, n_valid,
                                          output_scale);
      continue;
    }

    constexpr size_t M_TILE = 32;
    alignas(64) float acc_buf[M_TILE][32];
    for (size_t m0 = 0; m0 < M; m0 += M_TILE) {
      const size_t m_count = std::min<size_t>(M_TILE, M - m0);
      std::memset(acc_buf, 0, m_count * 32 * sizeof(float));
      const uint8_t* p_block = tile_ptr;
      for (size_t b = 0; b < B; ++b, p_block += block_bytes) {
        const auto* sp =
            reinterpret_cast<const __fp16*>(p_block + chunks * 512);
        const float16x8_t s0 = vld1q_f16(sp + 0), s1 = vld1q_f16(sp + 8);
        const float16x8_t s2 = vld1q_f16(sp + 16), s3 = vld1q_f16(sp + 24);
        const float32x4_t ws[8]{
            vcvt_f32_f16(vget_low_f16(s0)), vcvt_high_f32_f16(s0),
            vcvt_f32_f16(vget_low_f16(s1)), vcvt_high_f32_f16(s1),
            vcvt_f32_f16(vget_low_f16(s2)), vcvt_high_f32_f16(s2),
            vcvt_f32_f16(vget_low_f16(s3)), vcvt_high_f32_f16(s3),
        };
        float32x4_t wz[8];
        if constexpr (HAS_ZERO) {
          const auto* zp = sp + 32;
          const float16x8_t z0 = vld1q_f16(zp + 0), z1 = vld1q_f16(zp + 8);
          const float16x8_t z2 = vld1q_f16(zp + 16), z3 = vld1q_f16(zp + 24);
          wz[0] = vcvt_f32_f16(vget_low_f16(z0));
          wz[1] = vcvt_high_f32_f16(z0);
          wz[2] = vcvt_f32_f16(vget_low_f16(z1));
          wz[3] = vcvt_high_f32_f16(z1);
          wz[4] = vcvt_f32_f16(vget_low_f16(z2));
          wz[5] = vcvt_high_f32_f16(z2);
          wz[6] = vcvt_f32_f16(vget_low_f16(z3));
          wz[7] = vcvt_high_f32_f16(z3);
        }

        size_t m = 0;
        if constexpr (CHUNKS == 1) {
          for (; m + 2 <= m_count; m += 2) {
            int32x4_t d0[8], d1[8];
            for (size_t j = 0; j < 8; ++j) {
              d0[j] = vdupq_n_s32(0);
              d1[j] = vdupq_n_s32(0);
            }
            const uint8_t* p = p_block;
            const int8_t* qp0 = act_q + ((m0 + m) * B + b) * 32;
            const int8_t* qp1 = act_q + ((m0 + m + 1) * B + b) * 32;
            const int8x16_t qe0 = vld1q_s8(qp0 + 0), qo0 = vld1q_s8(qp0 + 16);
            const int8x16_t qe1 = vld1q_s8(qp1 + 0), qo1 = vld1q_s8(qp1 + 16);
            YNN_I4_DOT32x2_STEP(0, p, qe0, qo0, qe1, qo1, d0, d1);
            YNN_I4_DOT32x2_STEP(1, p, qe0, qo0, qe1, qo1, d0, d1);
            YNN_I4_DOT32x2_STEP(2, p, qe0, qo0, qe1, qo1, d0, d1);
            YNN_I4_DOT32x2_STEP(3, p, qe0, qo0, qe1, qo1, d0, d1);
            const float as0 = act_scale[(m0 + m) * B + b];
            const float as1 = act_scale[(m0 + m + 1) * B + b];
            for (size_t j = 0; j < 8; ++j) {
              float32x4_t a0 = vld1q_f32(acc_buf[m] + j * 4);
              float32x4_t a1 = vld1q_f32(acc_buf[m + 1] + j * 4);
              a0 = vfmaq_f32(a0, vcvtq_f32_s32(d0[j]), vmulq_n_f32(ws[j], as0));
              a1 = vfmaq_f32(a1, vcvtq_f32_s32(d1[j]), vmulq_n_f32(ws[j], as1));
              if constexpr (HAS_ZERO) {
                a0 = vfmaq_n_f32(a0, wz[j], act_bsum[(m0 + m) * B + b]);
                a1 = vfmaq_n_f32(a1, wz[j], act_bsum[(m0 + m + 1) * B + b]);
              }
              vst1q_f32(acc_buf[m] + j * 4, a0);
              vst1q_f32(acc_buf[m + 1] + j * 4, a1);
            }
          }
        } else {
          for (; m + 4 <= m_count; m += 4) {
            for (size_t half = 0; half < 2; ++half) {
              int32x4_t d[4][4];
              for (size_t r = 0; r < 4; ++r)
                for (size_t g = 0; g < 4; ++g) d[r][g] = vdupq_n_s32(0);
              const uint8_t* p = p_block + half * 64;
              for (size_t c = 0; c < chunks; ++c) {
                int8x16_t qe[4], qo[4];
                for (size_t r = 0; r < 4; ++r) {
                  const int8_t* qp =
                      act_q + ((m0 + m + r) * B + b) * G + c * 32;
                  qe[r] = vld1q_s8(qp + 0);
                  qo[r] = vld1q_s8(qp + 16);
                }
                YNN_I4_DOT16x4_STEP(0, p + 0 * 128, qe, qo, d);
                YNN_I4_DOT16x4_STEP(1, p + 1 * 128, qe, qo, d);
                YNN_I4_DOT16x4_STEP(2, p + 2 * 128, qe, qo, d);
                YNN_I4_DOT16x4_STEP(3, p + 3 * 128, qe, qo, d);
                p += 512;
              }
              for (size_t r = 0; r < 4; ++r) {
                const float as = act_scale[(m0 + m + r) * B + b];
                for (size_t g = 0; g < 4; ++g) {
                  const size_t j = half * 4 + g;
                  float32x4_t a = vld1q_f32(acc_buf[m + r] + j * 4);
                  a = vfmaq_f32(a, vcvtq_f32_s32(d[r][g]),
                                vmulq_n_f32(ws[j], as));
                  if constexpr (HAS_ZERO)
                    a = vfmaq_n_f32(a, wz[j], act_bsum[(m0 + m + r) * B + b]);
                  vst1q_f32(acc_buf[m + r] + j * 4, a);
                }
              }
            }
          }
        }
        for (; m < m_count; ++m) {
          int32x4_t d0[8];
          for (auto& v : d0) v = vdupq_n_s32(0);
          const uint8_t* p = p_block;
          const int8_t* qp0 = act_q + ((m0 + m) * B + b) * G;
          for (size_t c = 0; c < chunks; ++c, qp0 += 32) {
            const int8x16_t qe0 = vld1q_s8(qp0 + 0), qo0 = vld1q_s8(qp0 + 16);
            YNN_I4_DOT32_STEP(0, p, qe0, qo0, d0);
            YNN_I4_DOT32_STEP(1, p, qe0, qo0, d0);
            YNN_I4_DOT32_STEP(2, p, qe0, qo0, d0);
            YNN_I4_DOT32_STEP(3, p, qe0, qo0, d0);
          }
          const float as0 = act_scale[(m0 + m) * B + b];
          for (size_t j = 0; j < 8; ++j) {
            float32x4_t a0 = vld1q_f32(acc_buf[m] + j * 4);
            a0 = vfmaq_f32(a0, vcvtq_f32_s32(d0[j]), vmulq_n_f32(ws[j], as0));
            if constexpr (HAS_ZERO)
              a0 = vfmaq_n_f32(a0, wz[j], act_bsum[(m0 + m) * B + b]);
            vst1q_f32(acc_buf[m] + j * 4, a0);
          }
        }
      }
      for (size_t m = 0; m < m_count; ++m) {
        float32x4_t row_acc[8];
        for (size_t j = 0; j < 8; ++j)
          row_acc[j] = vld1q_f32(acc_buf[m] + j * 4);
        store_row32_f16(y + (m0 + m) * N + n0, row_acc, n_valid, output_scale);
      }
    }
  }
}

#undef YNN_I4_DOT32_STEP
#undef YNN_I4_DOT32x2_STEP
#undef YNN_I4_DOT16x4_STEP

#define YNN_I8_DOT32_STEP(STEP, P, QA, D)             \
  do {                                                \
    const int8x16_t r0 = vld1q_s8((P) + 0);           \
    const int8x16_t r1 = vld1q_s8((P) + 16);          \
    const int8x16_t r2 = vld1q_s8((P) + 32);          \
    const int8x16_t r3 = vld1q_s8((P) + 48);          \
    (D)[0] = vdotq_laneq_s32((D)[0], r0, (QA), STEP); \
    (D)[1] = vdotq_laneq_s32((D)[1], r1, (QA), STEP); \
    (D)[2] = vdotq_laneq_s32((D)[2], r2, (QA), STEP); \
    (D)[3] = vdotq_laneq_s32((D)[3], r3, (QA), STEP); \
    const int8x16_t r4 = vld1q_s8((P) + 64);          \
    const int8x16_t r5 = vld1q_s8((P) + 80);          \
    const int8x16_t r6 = vld1q_s8((P) + 96);          \
    const int8x16_t r7 = vld1q_s8((P) + 112);         \
    (P) += 128;                                       \
    (D)[4] = vdotq_laneq_s32((D)[4], r4, (QA), STEP); \
    (D)[5] = vdotq_laneq_s32((D)[5], r5, (QA), STEP); \
    (D)[6] = vdotq_laneq_s32((D)[6], r6, (QA), STEP); \
    (D)[7] = vdotq_laneq_s32((D)[7], r7, (QA), STEP); \
  } while (0)

#define YNN_I8_DOT16x4_STEP(STEP, P, QA, D)                      \
  do {                                                           \
    for (size_t g = 0; g < 4; ++g) {                             \
      const int8x16_t rw = vld1q_s8((P) + g * 16);               \
      (D)[0][g] = vdotq_laneq_s32((D)[0][g], rw, (QA)[0], STEP); \
      (D)[1][g] = vdotq_laneq_s32((D)[1][g], rw, (QA)[1], STEP); \
      (D)[2][g] = vdotq_laneq_s32((D)[2][g], rw, (QA)[2], STEP); \
      (D)[3][g] = vdotq_laneq_s32((D)[3][g], rw, (QA)[3], STEP); \
    }                                                            \
  } while (0)

__attribute__((target("dotprod"))) void eval_i8_tile_32x1(
    size_t B, size_t block_size, const int8_t* p, const int8_t* qp,
    const float* a_scale_base, const float* a_zp_base, float* y_base,
    size_t n_valid, float output_scale) noexcept {
  float32x4_t acc[8];
  for (auto& v : acc) v = vdupq_n_f32(0.0f);
  for (size_t b = 0; b < B; ++b) {
    int32x4_t d[8];
    for (auto& v : d) v = vdupq_n_s32(0);
    for (size_t k = 0; k < block_size; k += 16, qp += 16) {
      const int8x16_t qa = vld1q_s8(qp);
      YNN_I8_DOT32_STEP(0, p, qa, d);
      YNN_I8_DOT32_STEP(1, p, qa, d);
      YNN_I8_DOT32_STEP(2, p, qa, d);
      YNN_I8_DOT32_STEP(3, p, qa, d);
    }
    const auto* sp = reinterpret_cast<const float*>(p);
    p += 128;
    const auto* wsp = reinterpret_cast<const int32_t*>(p);
    p += 128;
    const int32_t zp =
        a_zp_base != nullptr ? static_cast<int32_t>(a_zp_base[b]) : 0;
    const float a_s = a_scale_base[b];
    for (size_t j = 0; j < 8; ++j) {
      const int32x4_t corr =
          zp != 0 ? vmlsq_n_s32(d[j], vld1q_s32(wsp + j * 4), zp) : d[j];
      acc[j] = vfmaq_f32(acc[j], vcvtq_f32_s32(corr),
                         vmulq_n_f32(vld1q_f32(sp + j * 4), a_s));
    }
  }
  store_row32_i8(y_base, acc, n_valid, output_scale);
}

__attribute__((target("dotprod"))) void dispatch_i8_tiles(
    size_t M, size_t N, size_t B, size_t block_size, float output_scale,
    const int8_t* packed_ptr, const int8_t* act_q, const float* act_scale,
    const float* act_zp, float* y, size_t t_begin, size_t t_end) noexcept {
  const size_t block_bytes = block_size * 32 + 256;
  const size_t tile_bytes = B * block_bytes;

  for (size_t tile = t_begin; tile < t_end; ++tile) {
    const size_t n0 = tile * 32;
    const size_t n_valid = std::min<size_t>(32, N - n0);
    const int8_t* tile_ptr = packed_ptr + tile * tile_bytes;

    if (M == 1) {
      eval_i8_tile_32x1(B, block_size, tile_ptr, act_q, act_scale, act_zp,
                        y + n0, n_valid, output_scale);
      continue;
    }

    constexpr size_t M_TILE = 32;
    constexpr size_t K_TILE = 512;
    alignas(64) float acc_buf[M_TILE][32];
    alignas(64) int32_t d_buf[M_TILE][32];
    for (size_t m0 = 0; m0 < M; m0 += M_TILE) {
      const size_t m_count = std::min<size_t>(M_TILE, M - m0);
      std::memset(acc_buf, 0, m_count * 32 * sizeof(float));
      const int8_t* p_block = tile_ptr;
      for (size_t b = 0; b < B; ++b, p_block += block_bytes) {
        std::memset(d_buf, 0, m_count * 32 * sizeof(int32_t));
        for (size_t k0 = 0; k0 < block_size; k0 += K_TILE) {
          const size_t k_len = std::min<size_t>(K_TILE, block_size - k0);
          const int8_t* p_sub = p_block + k0 * 32;
          size_t m = 0;
          for (; m + 4 <= m_count; m += 4) {
            for (size_t half = 0; half < 2; ++half) {
              int32x4_t d[4][4];
              for (size_t r = 0; r < 4; ++r)
                for (size_t g = 0; g < 4; ++g)
                  d[r][g] = vld1q_s32(d_buf[m + r] + (half * 4 + g) * 4);
              const int8_t* p = p_sub + half * 64;
              for (size_t k = 0; k < k_len; k += 16) {
                int8x16_t qa[4];
                for (size_t r = 0; r < 4; ++r)
                  qa[r] = vld1q_s8(act_q + ((m0 + m + r) * B + b) * block_size +
                                   k0 + k);
                YNN_I8_DOT16x4_STEP(0, p + 0 * 128, qa, d);
                YNN_I8_DOT16x4_STEP(1, p + 1 * 128, qa, d);
                YNN_I8_DOT16x4_STEP(2, p + 2 * 128, qa, d);
                YNN_I8_DOT16x4_STEP(3, p + 3 * 128, qa, d);
                p += 512;
              }
              for (size_t r = 0; r < 4; ++r)
                for (size_t g = 0; g < 4; ++g)
                  vst1q_s32(d_buf[m + r] + (half * 4 + g) * 4, d[r][g]);
            }
          }
          for (; m < m_count; ++m) {
            int32x4_t d0[8];
            for (size_t j = 0; j < 8; ++j) d0[j] = vld1q_s32(d_buf[m] + j * 4);
            const int8_t* p = p_sub;
            const int8_t* qp0 = act_q + ((m0 + m) * B + b) * block_size + k0;
            for (size_t k = 0; k < k_len; k += 16, qp0 += 16) {
              const int8x16_t qa0 = vld1q_s8(qp0);
              YNN_I8_DOT32_STEP(0, p, qa0, d0);
              YNN_I8_DOT32_STEP(1, p, qa0, d0);
              YNN_I8_DOT32_STEP(2, p, qa0, d0);
              YNN_I8_DOT32_STEP(3, p, qa0, d0);
            }
            for (size_t j = 0; j < 8; ++j) vst1q_s32(d_buf[m] + j * 4, d0[j]);
          }
        }
        const auto* sp =
            reinterpret_cast<const float*>(p_block + block_size * 32);
        const auto* wsp =
            reinterpret_cast<const int32_t*>(p_block + block_size * 32 + 128);
        float32x4_t ws[8];
        int32x4_t wsum[8];
        for (size_t j = 0; j < 8; ++j) {
          ws[j] = vld1q_f32(sp + j * 4);
          wsum[j] = vld1q_s32(wsp + j * 4);
        }
        for (size_t m = 0; m < m_count; ++m) {
          const float as = act_scale[(m0 + m) * B + b];
          const int32_t zp =
              act_zp != nullptr ? static_cast<int32_t>(act_zp[(m0 + m) * B + b])
                                : 0;
          for (size_t j = 0; j < 8; ++j) {
            float32x4_t a = vld1q_f32(acc_buf[m] + j * 4);
            const int32x4_t raw_d = vld1q_s32(d_buf[m] + j * 4);
            const int32x4_t corr =
                zp != 0 ? vmlsq_n_s32(raw_d, wsum[j], zp) : raw_d;
            a = vfmaq_f32(a, vcvtq_f32_s32(corr), vmulq_n_f32(ws[j], as));
            vst1q_f32(acc_buf[m] + j * 4, a);
          }
        }
      }
      for (size_t m = 0; m < m_count; ++m) {
        float32x4_t row_acc[8];
        for (size_t j = 0; j < 8; ++j)
          row_acc[j] = vld1q_f32(acc_buf[m] + j * 4);
        store_row32_i8(y + (m0 + m) * N + n0, row_acc, n_valid, output_scale);
      }
    }
  }
}

#undef YNN_I8_DOT32_STEP
#undef YNN_I8_DOT16x4_STEP
#endif

}  // namespace

bool is_blockwise_linear_supported(int32_t bits) {
#if defined(YNN_BLOCKWISE_ARM64)
  if (bits == 4)
    return is_arch_supported(arch_flag::neondot | arch_flag::neonfp16arith);
  if (bits == 8) return is_arch_supported(arch_flag::neondot);
#else
  (void)bits;
#endif
  return false;
}

size_t blockwise_linear_tile_count(size_t n) {
  return (n + NR_TILE - 1) / NR_TILE;
}

size_t packed_blockwise_linear_size(int32_t bits, size_t n, size_t k,
                                    size_t block_size, bool has_zero) {
  const size_t tiles = blockwise_linear_tile_count(n);
  const size_t B = k / block_size;
  if (bits == 4) {
    const size_t chunks = block_size / 32;
    const size_t block_bytes = chunks * 512 + 64 + (has_zero ? 64 : 0);
    return tiles * B * block_bytes;
  }
  if (bits == 8) {
    const size_t block_bytes = block_size * 32 + 256;
    return tiles * B * block_bytes;
  }
  return 0;
}

void pack_blockwise_linear_tiles(int32_t bits, size_t n, size_t k,
                                 size_t block_size, const void* weights,
                                 const void* scales, bool scales_fp16,
                                 const void* zeros, bool zeros_fp16,
                                 size_t tile_start, size_t tile_end,
                                 void* packed_tiles) {
#if defined(YNN_BLOCKWISE_ARM64)
  const size_t B = k / block_size;
  const size_t tiles = blockwise_linear_tile_count(n);
  tile_end = std::min(tile_end, tiles);
  auto* dst_all = static_cast<uint8_t*>(packed_tiles);
  const auto* s_f32 = scales_fp16 ? nullptr : static_cast<const float*>(scales);
  const auto* s_h16 =
      scales_fp16 ? static_cast<const __fp16*>(scales) : nullptr;

  if (bits == 4) {
    const size_t row_bytes = k / 2;
    const size_t chunks = block_size / 32;
    const bool has_zero = zeros != nullptr;
    const size_t block_bytes = chunks * 512 + 64 + (has_zero ? 64 : 0);
    const size_t tile_bytes = B * block_bytes;
    const auto* w_all = static_cast<const uint8_t*>(weights);
    const auto* z_f32 =
        (has_zero && !zeros_fp16) ? static_cast<const float*>(zeros) : nullptr;
    const auto* z_h16 =
        (has_zero && zeros_fp16) ? static_cast<const __fp16*>(zeros) : nullptr;

    for (size_t tile = tile_start; tile < tile_end; ++tile) {
      const size_t n0 = tile * 32;
      uint8_t* dst = dst_all + tile * tile_bytes;
      for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < chunks; ++c) {
          const size_t col_byte_base = (b * chunks + c) * 16;
          for (size_t step = 0; step < 4; ++step) {
            const size_t off = col_byte_base + step * 4;
            for (size_t r = 0; r < 32; ++r) {
              const size_t row = n0 + r;
              uint32_t word = 0;
              if (row < n)
                std::memcpy(&word, w_all + row * row_bytes + off,
                            sizeof(uint32_t));
              std::memcpy(dst + r * 4, &word, sizeof(uint32_t));
            }
            dst += 128;
          }
        }
        auto* dst_s = reinterpret_cast<__fp16*>(dst);
        for (size_t r = 0; r < 32; ++r) {
          const size_t row = n0 + r;
          const size_t idx = row * B + b;
          dst_s[r] = row < n ? (scales_fp16 ? s_h16[idx]
                                            : static_cast<__fp16>(s_f32[idx]))
                             : static_cast<__fp16>(0.0f);
        }
        dst += 64;
        if (has_zero) {
          auto* dst_z = reinterpret_cast<__fp16*>(dst);
          for (size_t r = 0; r < 32; ++r) {
            const size_t row = n0 + r;
            const size_t idx = row * B + b;
            dst_z[r] = row < n ? (zeros_fp16 ? z_h16[idx]
                                             : static_cast<__fp16>(z_f32[idx]))
                               : static_cast<__fp16>(0.0f);
          }
          dst += 64;
        }
      }
    }
    return;
  }

  if (bits == 8) {
    const size_t block_bytes = block_size * 32 + 256;
    const size_t tile_bytes = B * block_bytes;
    const auto* w_all = static_cast<const int8_t*>(weights);
    for (size_t tile = tile_start; tile < tile_end; ++tile) {
      const size_t n0 = tile * 32;
      uint8_t* dst = dst_all + tile * tile_bytes;
      for (size_t b = 0; b < B; ++b) {
        const size_t col_base = b * block_size;
        int32_t wsum[32] = {};
        for (size_t k_idx = 0; k_idx < block_size; k_idx += 4) {
          for (size_t r = 0; r < 32; ++r) {
            const size_t row = n0 + r;
            int32_t word = 0;
            if (row < n) {
              const int8_t* wp = w_all + row * k + col_base + k_idx;
              std::memcpy(&word, wp, sizeof(int32_t));
              wsum[r] +=
                  static_cast<int32_t>(wp[0]) + static_cast<int32_t>(wp[1]) +
                  static_cast<int32_t>(wp[2]) + static_cast<int32_t>(wp[3]);
            }
            std::memcpy(dst + r * 4, &word, sizeof(int32_t));
          }
          dst += 128;
        }
        auto* dst_s = reinterpret_cast<float*>(dst);
        for (size_t r = 0; r < 32; ++r) {
          const size_t row = n0 + r;
          const size_t idx = row * B + b;
          dst_s[r] = row < n ? (scales_fp16 ? static_cast<float>(s_h16[idx])
                                            : s_f32[idx])
                             : 0.0f;
        }
        dst += 128;
        std::memcpy(dst, wsum, 32 * sizeof(int32_t));
        dst += 128;
      }
    }
  }
#else
  (void)bits;
  (void)n;
  (void)k;
  (void)block_size;
  (void)weights;
  (void)scales;
  (void)scales_fp16;
  (void)zeros;
  (void)zeros_fp16;
  (void)tile_start;
  (void)tile_end;
  (void)packed_tiles;
#endif
}

void quantize_blockwise_activations_f32(int32_t bits, size_t m, size_t k,
                                        size_t block_size, bool compute_bsum,
                                        const float* input,
                                        int8_t* quantized_input,
                                        float* input_scales, float* input_bsums,
                                        size_t row_start, size_t row_end) {
#if defined(YNN_BLOCKWISE_ARM64)
  row_end = std::min(row_end, m);
  const size_t B = k / block_size;
  if (bits == 4) {
    const size_t chunks = block_size / 32;
    for (size_t row = row_start; row < row_end; ++row) {
      for (size_t b = 0; b < B; ++b) {
        const float* x_block = input + (row * B + b) * block_size;
        float32x4_t vsum0 = vdupq_n_f32(0.0f), vsum1 = vdupq_n_f32(0.0f);
        float32x4_t vmax0 = vdupq_n_f32(0.0f), vmax1 = vdupq_n_f32(0.0f);
        for (size_t c = 0; c < chunks; ++c) {
          const float* xp = x_block + c * 32;
          const float32x4_t x0 = vld1q_f32(xp + 0), x1 = vld1q_f32(xp + 4);
          const float32x4_t x2 = vld1q_f32(xp + 8), x3 = vld1q_f32(xp + 12);
          const float32x4_t x4 = vld1q_f32(xp + 16), x5 = vld1q_f32(xp + 20);
          const float32x4_t x6 = vld1q_f32(xp + 24), x7 = vld1q_f32(xp + 28);
          if (compute_bsum) {
            vsum0 = vaddq_f32(vsum0,
                              vaddq_f32(vaddq_f32(x0, x1), vaddq_f32(x2, x3)));
            vsum1 = vaddq_f32(vsum1,
                              vaddq_f32(vaddq_f32(x4, x5), vaddq_f32(x6, x7)));
          }
          vmax0 = vmaxq_f32(vmax0,
                            vmaxq_f32(vmaxq_f32(vabsq_f32(x0), vabsq_f32(x1)),
                                      vmaxq_f32(vabsq_f32(x2), vabsq_f32(x3))));
          vmax1 = vmaxq_f32(vmax1,
                            vmaxq_f32(vmaxq_f32(vabsq_f32(x4), vabsq_f32(x5)),
                                      vmaxq_f32(vabsq_f32(x6), vabsq_f32(x7))));
        }
        const float amax = vmaxvq_f32(vmaxq_f32(vmax0, vmax1));
        const float scale = amax / 127.0f;
        const float inv_scale = amax > 0.0f ? 127.0f / amax : 0.0f;
        input_scales[row * B + b] = scale;
        if (compute_bsum)
          input_bsums[row * B + b] = vaddvq_f32(vaddq_f32(vsum0, vsum1));

        const float32x4_t vinv = vdupq_n_f32(inv_scale);
        for (size_t c = 0; c < chunks; ++c) {
          const float* xp = x_block + c * 32;
          const int32x4_t i0 =
              vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 0), vinv));
          const int32x4_t i1 =
              vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 4), vinv));
          const int32x4_t i2 =
              vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 8), vinv));
          const int32x4_t i3 =
              vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 12), vinv));
          const int32x4_t i4 =
              vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 16), vinv));
          const int32x4_t i5 =
              vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 20), vinv));
          const int32x4_t i6 =
              vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 24), vinv));
          const int32x4_t i7 =
              vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 28), vinv));
          const int16x8_t s0 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
          const int16x8_t s1 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
          const int16x8_t s2 = vcombine_s16(vqmovn_s32(i4), vqmovn_s32(i5));
          const int16x8_t s3 = vcombine_s16(vqmovn_s32(i6), vqmovn_s32(i7));
          const int8x16_t q0 = vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1));
          const int8x16_t q1 = vcombine_s8(vqmovn_s16(s2), vqmovn_s16(s3));
          int8_t* qp = quantized_input + (row * B + b) * block_size + c * 32;
          vst1q_s8(qp + 0, vuzp1q_s8(q0, q1));
          vst1q_s8(qp + 16, vuzp2q_s8(q0, q1));
        }
      }
    }
    return;
  }

  if (bits == 8) {
    for (size_t row = row_start; row < row_end; ++row) {
      for (size_t b = 0; b < B; ++b) {
        const float* x_block = input + (row * B + b) * block_size;
        if (compute_bsum && input_bsums != nullptr) {
          float32x4_t vmin0 = vdupq_n_f32(0.0f), vmin1 = vdupq_n_f32(0.0f);
          float32x4_t vmax0 = vdupq_n_f32(0.0f), vmax1 = vdupq_n_f32(0.0f);
          size_t c = 0;
          for (; c + 32 <= block_size; c += 32) {
            const float* xp = x_block + c;
            const float32x4_t x0 = vld1q_f32(xp + 0), x1 = vld1q_f32(xp + 4);
            const float32x4_t x2 = vld1q_f32(xp + 8), x3 = vld1q_f32(xp + 12);
            const float32x4_t x4 = vld1q_f32(xp + 16), x5 = vld1q_f32(xp + 20);
            const float32x4_t x6 = vld1q_f32(xp + 24), x7 = vld1q_f32(xp + 28);
            vmin0 = vminq_f32(vmin0,
                              vminq_f32(vminq_f32(x0, x1), vminq_f32(x2, x3)));
            vmin1 = vminq_f32(vmin1,
                              vminq_f32(vminq_f32(x4, x5), vminq_f32(x6, x7)));
            vmax0 = vmaxq_f32(vmax0,
                              vmaxq_f32(vmaxq_f32(x0, x1), vmaxq_f32(x2, x3)));
            vmax1 = vmaxq_f32(vmax1,
                              vmaxq_f32(vmaxq_f32(x4, x5), vmaxq_f32(x6, x7)));
          }
          for (; c < block_size; c += 16) {
            const float* xp = x_block + c;
            const float32x4_t x0 = vld1q_f32(xp + 0), x1 = vld1q_f32(xp + 4);
            const float32x4_t x2 = vld1q_f32(xp + 8), x3 = vld1q_f32(xp + 12);
            vmin0 = vminq_f32(vmin0,
                              vminq_f32(vminq_f32(x0, x1), vminq_f32(x2, x3)));
            vmax0 = vmaxq_f32(vmax0,
                              vmaxq_f32(vmaxq_f32(x0, x1), vmaxq_f32(x2, x3)));
          }
          const float rmin = vminvq_f32(vminq_f32(vmin0, vmin1));
          const float rmax = vmaxvq_f32(vmaxq_f32(vmax0, vmax1));
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
            zp = static_cast<int32_t>(
                std::rint(std::clamp(fzp, -128.0f, 127.0f)));
          }
          input_scales[row * B + b] = scale;
          input_bsums[row * B + b] = static_cast<float>(zp);
          const float32x4_t vinv = vdupq_n_f32(inv_scale);
          const int32x4_t vzp = vdupq_n_s32(zp);
          int8_t* qp = quantized_input + (row * B + b) * block_size;
          for (size_t col = 0; col < block_size; col += 16) {
            const float* xp = x_block + col;
            const int32x4_t i0 = vaddq_s32(
                vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xp + 0), vinv)), vzp);
            const int32x4_t i1 = vaddq_s32(
                vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xp + 4), vinv)), vzp);
            const int32x4_t i2 = vaddq_s32(
                vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xp + 8), vinv)), vzp);
            const int32x4_t i3 = vaddq_s32(
                vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xp + 12), vinv)), vzp);
            const int16x8_t s0 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
            const int16x8_t s1 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
            vst1q_s8(qp + col, vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1)));
          }
          continue;
        }
        float32x4_t vmax0 = vdupq_n_f32(0.0f), vmax1 = vdupq_n_f32(0.0f);
        size_t c = 0;
        for (; c + 32 <= block_size; c += 32) {
          const float* xp = x_block + c;
          vmax0 = vmaxq_f32(
              vmax0, vmaxq_f32(vmaxq_f32(vabsq_f32(vld1q_f32(xp + 0)),
                                         vabsq_f32(vld1q_f32(xp + 4))),
                               vmaxq_f32(vabsq_f32(vld1q_f32(xp + 8)),
                                         vabsq_f32(vld1q_f32(xp + 12)))));
          vmax1 = vmaxq_f32(
              vmax1, vmaxq_f32(vmaxq_f32(vabsq_f32(vld1q_f32(xp + 16)),
                                         vabsq_f32(vld1q_f32(xp + 20))),
                               vmaxq_f32(vabsq_f32(vld1q_f32(xp + 24)),
                                         vabsq_f32(vld1q_f32(xp + 28)))));
        }
        for (; c < block_size; c += 16) {
          const float* xp = x_block + c;
          vmax0 = vmaxq_f32(
              vmax0, vmaxq_f32(vmaxq_f32(vabsq_f32(vld1q_f32(xp + 0)),
                                         vabsq_f32(vld1q_f32(xp + 4))),
                               vmaxq_f32(vabsq_f32(vld1q_f32(xp + 8)),
                                         vabsq_f32(vld1q_f32(xp + 12)))));
        }
        const float amax = vmaxvq_f32(vmaxq_f32(vmax0, vmax1));
        const float scale = amax / 127.0f;
        const float inv_scale = amax > 0.0f ? 127.0f / amax : 0.0f;
        input_scales[row * B + b] = scale;

        const float32x4_t vinv = vdupq_n_f32(inv_scale);
        int8_t* qp = quantized_input + (row * B + b) * block_size;
        for (size_t col = 0; col < block_size; col += 16) {
          const float* xp = x_block + col;
          const int32x4_t i0 =
              vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 0), vinv));
          const int32x4_t i1 =
              vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 4), vinv));
          const int32x4_t i2 =
              vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 8), vinv));
          const int32x4_t i3 =
              vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 12), vinv));
          const int16x8_t s0 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
          const int16x8_t s1 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
          vst1q_s8(qp + col, vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1)));
        }
      }
    }
  }
#else
  (void)bits;
  (void)m;
  (void)k;
  (void)block_size;
  (void)compute_bsum;
  (void)input;
  (void)quantized_input;
  (void)input_scales;
  (void)input_bsums;
  (void)row_start;
  (void)row_end;
#endif
}

void blockwise_linear_f32(int32_t bits, size_t m, size_t n, size_t k,
                          size_t block_size, bool has_zero, float output_scale,
                          const void* packed_tiles,
                          const int8_t* quantized_input,
                          const float* input_scales, const float* input_bsums,
                          float* output, size_t tile_start, size_t tile_end) {
#if defined(YNN_BLOCKWISE_ARM64)
  const size_t tiles = blockwise_linear_tile_count(n);
  tile_end = std::min(tile_end, tiles);
  if (tile_start >= tile_end) return;
  const size_t B = k / block_size;

  if (bits == 4) {
    const auto* ptr = static_cast<const uint8_t*>(packed_tiles);
    const size_t chunks = block_size / 32;
    if (has_zero) {
      if (chunks == 1) {
        dispatch_i4_tiles<1, true>(m, n, B, block_size, output_scale, ptr,
                                   quantized_input, input_scales, input_bsums,
                                   output, tile_start, tile_end);
      } else if (chunks == 2) {
        dispatch_i4_tiles<2, true>(m, n, B, block_size, output_scale, ptr,
                                   quantized_input, input_scales, input_bsums,
                                   output, tile_start, tile_end);
      } else if (chunks == 4) {
        dispatch_i4_tiles<4, true>(m, n, B, block_size, output_scale, ptr,
                                   quantized_input, input_scales, input_bsums,
                                   output, tile_start, tile_end);
      } else {
        dispatch_i4_tiles<0, true>(m, n, B, block_size, output_scale, ptr,
                                   quantized_input, input_scales, input_bsums,
                                   output, tile_start, tile_end);
      }
    } else {
      if (chunks == 1) {
        dispatch_i4_tiles<1, false>(m, n, B, block_size, output_scale, ptr,
                                    quantized_input, input_scales, input_bsums,
                                    output, tile_start, tile_end);
      } else if (chunks == 2) {
        dispatch_i4_tiles<2, false>(m, n, B, block_size, output_scale, ptr,
                                    quantized_input, input_scales, input_bsums,
                                    output, tile_start, tile_end);
      } else if (chunks == 4) {
        dispatch_i4_tiles<4, false>(m, n, B, block_size, output_scale, ptr,
                                    quantized_input, input_scales, input_bsums,
                                    output, tile_start, tile_end);
      } else {
        dispatch_i4_tiles<0, false>(m, n, B, block_size, output_scale, ptr,
                                    quantized_input, input_scales, input_bsums,
                                    output, tile_start, tile_end);
      }
    }
    return;
  }

  if (bits == 8) {
    const auto* ptr = static_cast<const int8_t*>(packed_tiles);
    dispatch_i8_tiles(m, n, B, block_size, output_scale, ptr, quantized_input,
                      input_scales, input_bsums, output, tile_start, tile_end);
  }
#else
  (void)bits;
  (void)m;
  (void)n;
  (void)k;
  (void)block_size;
  (void)has_zero;
  (void)output_scale;
  (void)packed_tiles;
  (void)quantized_input;
  (void)input_scales;
  (void)input_bsums;
  (void)output;
  (void)tile_start;
  (void)tile_end;
#endif
}

}  // namespace ynn
