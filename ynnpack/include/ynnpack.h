// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_INCLUDE_YNNPACK_H_
#define XNNPACK_YNNPACK_INCLUDE_YNNPACK_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// A value ID that is never valid.
#define YNN_INVALID_VALUE_ID UINT32_MAX

// The most dimensions that can appear in a value.
#define YNN_MAX_TENSOR_RANK 8

// This flag indicates that YNNPACK should attempt to produce numerically
// consistent results from a specific build of YNNPACK. This causes YNNPACK to
// avoid using faster codepaths that are numerically inconsistent with any
// other codepath that could be used in the same compiled YNNPACK library.
#define YNN_FLAG_CONSISTENT_ARITHMETIC (1 << 1)

// Disallows optimizations that result in more precision than the graph
// specified.
#define YNN_FLAG_NO_EXCESS_PRECISION (1 << 2)

// Allows fast, lower-accuracy approximations for transcendental functions.
#define YNN_FLAG_FAST_MATH (1 << 3)

#ifdef __GNUC__
#define YNN_DEPRECATED __attribute__((deprecated))
#else
#define YNN_DEPRECATED
#endif

enum ynn_status {
  ynn_status_success = 0,
  ynn_status_error = 1,
  ynn_status_invalid_parameter = 2,
  ynn_status_unsupported_parameter = 3,
  ynn_status_deprecated = 4,
};

// Describes a type for a value.
enum ynn_type {
  ynn_type_invalid = -1,

  ynn_type_int2 = 0,
  ynn_type_uint2 = 1,
  ynn_type_int4 = 2,
  ynn_type_uint4 = 3,
  ynn_type_int8 = 4,
  ynn_type_uint8 = 5,
  ynn_type_int32 = 6,
  ynn_type_fp16 = 7,
  ynn_type_bf16 = 8,
  ynn_type_fp32 = 9,
  ynn_type_fp64 = 10,
  ynn_type_fp8_e5m2 = 11,
  ynn_type_fp8_e4m3 = 12,
};

enum ynn_unary_operator {
  ynn_unary_invalid = 0,

  ynn_unary_abs = 1,
  ynn_unary_ceil = 2,
  ynn_unary_convert = 3,
  ynn_unary_cos = 4,
  ynn_unary_cbrt = 5,
  ynn_unary_erf = 6,
  ynn_unary_exp = 7,
  ynn_unary_expm1 = 8,
  ynn_unary_floor = 9,
  ynn_unary_hardswish = 10,
  ynn_unary_log = 11,
  ynn_unary_log1p = 12,
  ynn_unary_negate = 13,
  ynn_unary_rsqrt = 14,
  ynn_unary_round = 15,
  ynn_unary_sigmoid = 16,
  ynn_unary_sign = 17,
  ynn_unary_sin = 18,
  ynn_unary_square = 19,
  ynn_unary_sqrt = 20,
  ynn_unary_tanh = 21,
  ynn_unary_poly3 = 22,
  ynn_unary_round_to_bf16 = 23,
  ynn_unary_approx_erf = 24,
  ynn_unary_approx_tanh = 25,
  ynn_unary_tan = 26,

  // Internal use only
  ynn_unary_requantize_to_uint8,

  // For backwards compatibility
  ynn_unary_cosine = ynn_unary_cos,
  ynn_unary_sine = ynn_unary_sin,
  ynn_unary_tangent = ynn_unary_tan,
  ynn_unary_square_root = ynn_unary_sqrt,
  ynn_unary_reciprocal_square_root = ynn_unary_rsqrt,
  ynn_unary_cube_root = ynn_unary_cbrt,
};

enum ynn_binary_operator {
  ynn_binary_invalid = 0,

  ynn_binary_add = 1,
  ynn_binary_copysign = 2,
  ynn_binary_divide = 3,
  ynn_binary_leaky_relu = 4,  // computes a < 0 ? a * b : a
  ynn_binary_max = 5,
  ynn_binary_min = 6,
  ynn_binary_multiply = 7,
  ynn_binary_pow = 8,
  ynn_binary_squared_difference = 9,
  ynn_binary_subtract = 10,

  // Internal use only
  ynn_binary_exp_subtract,
};

enum ynn_reduce_operator {
  ynn_reduce_invalid = 0,

  ynn_reduce_max = 1,
  ynn_reduce_min = 2,
  ynn_reduce_min_max = 3,
  ynn_reduce_sum = 4,
  ynn_reduce_sum_squared = 5,
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_YNNPACK_INCLUDE_YNNPACK_H_
