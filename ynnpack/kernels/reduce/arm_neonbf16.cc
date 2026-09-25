// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>

#include <cstddef>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/simd/arm_vec128.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum.h"

namespace ynn {

namespace simd {

static f32x8 reduce_add(
    f32x8 a, bf16x8 b, identity /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  bfloat16x8_t one = vreinterpretq_bf16_u16(vdupq_n_u16(0x3F80));
  uint16x8x2_t zipped = vzipq_u16(b.v, vdupq_n_u16(0x0000));
  bfloat16x8_t evens = vreinterpretq_bf16_u16(zipped.val[0]);
  bfloat16x8_t odds = vreinterpretq_bf16_u16(zipped.val[1]);
  return a + concat(f32x4{vbfdotq_f32(vdupq_n_f32(0), evens, one)},
                    f32x4{vbfdotq_f32(vdupq_n_f32(0), odds, one)});
}

static f32x16 reduce_add(f32x16 a, vec<bfloat16, 16> b, identity map_fn,
                         std::integral_constant<size_t, 1> horizontal_factor) {
  return concat(reduce_add(extract<0>(a, f32x8::N), extract<0>(b, bf16x8::N),
                           map_fn, horizontal_factor),
                reduce_add(extract<1>(a, f32x8::N), extract<1>(b, bf16x8::N),
                           map_fn, horizontal_factor));
}

static f32x8 reduce_add(
    f32x8 a, bf16x8 b, square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  uint16x8x2_t zipped = vzipq_u16(b.v, vdupq_n_u16(0x0000));
  bfloat16x8_t evens = vreinterpretq_bf16_u16(zipped.val[0]);
  bfloat16x8_t odds = vreinterpretq_bf16_u16(zipped.val[1]);
  return a + concat(f32x4{vbfdotq_f32(vdupq_n_f32(0), evens, evens)},
                    f32x4{vbfdotq_f32(vdupq_n_f32(0), odds, odds)});
}

static f32x16 reduce_add(f32x16 a, vec<bfloat16, 16> b, square map_fn,
                         std::integral_constant<size_t, 1> horizontal_factor) {
  return concat(reduce_add(extract<0>(a, f32x8::N), extract<0>(b, bf16x8::N),
                           map_fn, horizontal_factor),
                reduce_add(extract<1>(a, f32x8::N), extract<1>(b, bf16x8::N),
                           map_fn, horizontal_factor));
}

}  // namespace simd

using simd::bf16x8;
using simd::f32x4;
using simd::f32x8;

SUM_FLOAT_K1_KERNEL(sum_k1_bf16_fp32_neonbf16, bfloat16, float, 0, 1, identity);
SUM_FLOAT_KN_KERNEL(sum_kn_bf16_fp32_neonbf16, bfloat16, float, 8, identity);

SUM_FLOAT_K1_KERNEL(sum_squared_k1_bf16_fp32_neonbf16, bfloat16, float, 0, 1,
                    square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_bf16_fp32_neonbf16, bfloat16, float, 8,
                    square);

}  // namespace ynn
