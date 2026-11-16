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
#include <type_traits>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/fp8.h"
#include "ynnpack/base/half.h"
#include "ynnpack/composites/composites.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/unary/unary.h"

namespace ynn {
namespace {

inline float load_elem_f32(ynn_type dt, const void* ptr, size_t idx) {
    switch (dt) {
        case ynn_type_fp32:
            return static_cast<const float*>(ptr)[idx];
        case ynn_type_bf16:
            return static_cast<float>(static_cast<const bfloat16*>(ptr)[idx]);
        case ynn_type_fp16:
            return static_cast<float>(static_cast<const half*>(ptr)[idx]);
        case ynn_type_fp8_e4m3:
            return static_cast<float>(static_cast<const fp8_e4m3*>(ptr)[idx]);
        case ynn_type_fp8_e5m2:
            return static_cast<float>(static_cast<const fp8_e5m2*>(ptr)[idx]);
        case ynn_type_int32:
            return static_cast<float>(static_cast<const int32_t*>(ptr)[idx]);
        case ynn_type_int8:
            return static_cast<float>(static_cast<const int8_t*>(ptr)[idx]);
        case ynn_type_uint8:
            return static_cast<float>(static_cast<const uint8_t*>(ptr)[idx]);
        default:
            return 0.0f;
    }
}

template <typename IntT>
inline IntT clamp_round_to_int(float v) {
    const float r = std::nearbyint(v);
    constexpr float lo = static_cast<float>(std::numeric_limits<IntT>::min());
    constexpr float hi = static_cast<float>(std::numeric_limits<IntT>::max());
    if (!(r >= lo)) return std::numeric_limits<IntT>::min();
    if (r >= hi) return std::numeric_limits<IntT>::max();
    return static_cast<IntT>(r);
}

inline void store_elem_f32(ynn_type dt, void* ptr, size_t idx, float v) {
    switch (dt) {
        case ynn_type_fp32:
            static_cast<float*>(ptr)[idx] = v;
            break;
        case ynn_type_bf16:
            static_cast<bfloat16*>(ptr)[idx] = bfloat16(v);
            break;
        case ynn_type_fp16:
            static_cast<half*>(ptr)[idx] = half(v);
            break;
        case ynn_type_fp8_e4m3:
            static_cast<fp8_e4m3*>(ptr)[idx] = fp8_e4m3(v);
            break;
        case ynn_type_fp8_e5m2:
            static_cast<fp8_e5m2*>(ptr)[idx] = fp8_e5m2(v);
            break;
        case ynn_type_int32:
            static_cast<int32_t*>(ptr)[idx] = clamp_round_to_int<int32_t>(v);
            break;
        case ynn_type_int8:
            static_cast<int8_t*>(ptr)[idx] = clamp_round_to_int<int8_t>(v);
            break;
        case ynn_type_uint8:
            static_cast<uint8_t*>(ptr)[idx] = clamp_round_to_int<uint8_t>(v);
            break;
        default:
            break;
    }
}

} // namespace

void layer_norm_f32(size_t rows, size_t width, float epsilon, const float* input, const float* residual_in,
                    const float* weight, const float* bias, float* output_norm, float* output_residual,
                    size_t row_start, size_t row_end) {
    row_end = std::min(row_end, rows);
    if (width == 0 || row_start >= row_end) return;
    const float inv_width = 1.0f / static_cast<float>(width);

    for (size_t r = row_start; r < row_end; ++r) {
        const float* x = input + r * width;
        const float* res = residual_in != nullptr ? residual_in + r * width : nullptr;
        float* y = output_norm + r * width;
        float* y_res = output_residual != nullptr ? output_residual + r * width : nullptr;

        float sum = 0.0f;
#if defined(__ARM_NEON)
        float32x4_t vsum = vdupq_n_f32(0.0f);
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
#else
        for (size_t i = 0; i < width; ++i) {
            const float v = res != nullptr ? (x[i] + res[i]) : x[i];
            if (y_res != nullptr) y_res[i] = v;
            sum += v;
        }
#endif
        const float mean = sum * inv_width;

        float var_sum = 0.0f;
        const float* src = y_res != nullptr ? y_res : x;
#if defined(__ARM_NEON)
        float32x4_t vvsum = vdupq_n_f32(0.0f);
        const float32x4_t vmean = vdupq_n_f32(mean);
        size_t j = 0;
        for (; j + 4 <= width; j += 4) {
            const float32x4_t v = res != nullptr && y_res == nullptr ? vaddq_f32(vld1q_f32(x + j), vld1q_f32(res + j))
                                                                     : vld1q_f32(src + j);
            const float32x4_t d = vsubq_f32(v, vmean);
            vvsum = vfmaq_f32(vvsum, d, d);
        }
        var_sum = vaddvq_f32(vvsum);
        for (; j < width; ++j) {
            const float v = (res != nullptr && y_res == nullptr) ? (x[j] + res[j]) : src[j];
            const float d = v - mean;
            var_sum += d * d;
        }
#else
        for (size_t j = 0; j < width; ++j) {
            const float v = (res != nullptr && y_res == nullptr) ? (x[j] + res[j]) : src[j];
            const float d = v - mean;
            var_sum += d * d;
        }
#endif
        const float inv_std = 1.0f / std::sqrt(var_sum * inv_width + epsilon);

#if defined(__ARM_NEON)
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
#else
        for (size_t k = 0; k < width; ++k) {
            const float v = (res != nullptr && y_res == nullptr) ? (x[k] + res[k]) : src[k];
            y[k] = (v - mean) * inv_std * weight[k] + bias[k];
        }
#endif
    }
}

void softmax_f32(size_t rows, size_t width, bool logarithmic, const float* input, float* output, size_t row_start,
                 size_t row_end) {
    row_end = std::min(row_end, rows);
    if (width == 0 || row_start >= row_end) return;

    for (size_t r = row_start; r < row_end; ++r) {
        const float* x = input + r * width;
        float* y = output + r * width;

        float max_val = -std::numeric_limits<float>::infinity();
#if defined(__ARM_NEON)
        float32x4_t vmax = vdupq_n_f32(-std::numeric_limits<float>::infinity());
        size_t i = 0;
        for (; i + 4 <= width; i += 4) vmax = vmaxq_f32(vmax, vld1q_f32(x + i));
        max_val = vmaxvq_f32(vmax);
        for (; i < width; ++i) max_val = std::max(max_val, x[i]);
#else
        for (size_t i = 0; i < width; ++i) max_val = std::max(max_val, x[i]);
#endif
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
#if defined(__ARM_NEON)
            const float32x4_t vinv = vdupq_n_f32(inv_sum);
            size_t j = 0;
            for (; j + 4 <= width; j += 4) vst1q_f32(y + j, vmulq_f32(vld1q_f32(y + j), vinv));
            for (; j < width; ++j) y[j] *= inv_sum;
#else
            for (size_t j = 0; j < width; ++j) y[j] *= inv_sum;
#endif
        }
    }
}

void gelu_f32(size_t count, bool approximate, const float* input, float* output, size_t start, size_t end) {
    end = std::min(end, count);
    if (start >= end) return;
    if (approximate) {
        static const auto tanh_kernel = get_unary_kernel(ynn_unary_tanh, ynn_type_fp32, ynn_type_fp32);
        static const auto tanh_params = get_unary_params(ynn_unary_tanh);
        constexpr float kSqrt2OverPi = 0.79788456080286535588f;
        constexpr size_t kChunk = 1024;
        alignas(64) float inner_buf[kChunk];
        for (size_t offset = start; offset < end; offset += kChunk) {
            const size_t n = std::min(kChunk, end - offset);
            const float* x = input + offset;
            float* y = output + offset;
            if (tanh_kernel) {
                {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
                    for (size_t i = 0; i < n; ++i) {
                        const float v = x[i];
                        inner_buf[i] = (v + (v * v * v) * 0.044715f) * kSqrt2OverPi;
                    }
                }
                tanh_kernel(1, n, n * sizeof(float), inner_buf, n * sizeof(float), inner_buf, &tanh_params);
                {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
                    for (size_t i = 0; i < n; ++i) y[i] = (x[i] * 0.5f) * (1.0f + inner_buf[i]);
                }
            } else {
                for (size_t i = 0; i < n; ++i) {
                    const float v = x[i];
                    y[i] = 0.5f * v * (1.0f + std::tanh(kSqrt2OverPi * (v + 0.044715f * v * v * v)));
                }
            }
        }
    } else {
        constexpr float kInvSqrt2 = 0.70710678118654752440f;
        for (size_t i = start; i < end; ++i) {
            const float x = input[i];
            output[i] = x * (0.5f * std::erf(x * kInvSqrt2) + 0.5f);
        }
    }
}

void tanh_f32(size_t count, const float* input, float* output, size_t start, size_t end) {
    end = std::min(end, count);
    if (start >= end) return;
    static const auto tanh_kernel = get_unary_kernel(ynn_unary_tanh, ynn_type_fp32, ynn_type_fp32);
    static const auto tanh_params = get_unary_params(ynn_unary_tanh);
    if (tanh_kernel) {
        const size_t n = end - start;
        tanh_kernel(1, n, n * sizeof(float), input + start, n * sizeof(float), output + start, &tanh_params);
        return;
    }
    for (size_t i = start; i < end; ++i) output[i] = std::tanh(input[i]);
}

void static_round_f32(size_t count, float scale, const float* input, float* output, size_t start, size_t end) {
    end = std::min(end, count);
    const float inv_scale = 1.0f / scale;
    for (size_t i = start; i < end; ++i) {
        const float q = std::nearbyint(input[i] * inv_scale);
        output[i] = std::clamp(q, -128.0f, 127.0f) * scale;
    }
}

void cast_elements(ynn_type src_type, ynn_type dst_type, size_t count, const void* src, void* dst, size_t start,
                   size_t end) {
    end = std::min(end, count);
    if (start >= end) return;
    if (src_type == dst_type) {
        size_t elem_bytes = 0;
        switch (src_type) {
            case ynn_type_fp32:
            case ynn_type_int32:
                elem_bytes = 4;
                break;
            case ynn_type_bf16:
            case ynn_type_fp16:
                elem_bytes = 2;
                break;
            case ynn_type_fp8_e4m3:
            case ynn_type_fp8_e5m2:
            case ynn_type_int8:
            case ynn_type_uint8:
                elem_bytes = 1;
                break;
            default:
                return;
        }
        std::memcpy(static_cast<uint8_t*>(dst) + start * elem_bytes,
                    static_cast<const uint8_t*>(src) + start * elem_bytes, (end - start) * elem_bytes);
        return;
    }
    if (src_type == ynn_type_fp32 && dst_type == ynn_type_bf16) {
        convert_f32_to_bf16_rows(1, end - start, static_cast<const float*>(src) + start, end - start,
                                 static_cast<uint16_t*>(dst) + start, end - start, 0, 1);
        return;
    }
    if (src_type == ynn_type_bf16 && dst_type == ynn_type_fp32) {
        convert_bf16_to_f32_rows(1, end - start, static_cast<const uint16_t*>(src) + start, end - start,
                                 static_cast<float*>(dst) + start, end - start, 0, 1);
        return;
    }
    for (size_t i = start; i < end; ++i) {
        store_elem_f32(dst_type, dst, i, load_elem_f32(src_type, src, i));
    }
}

void broadcast_binary_f32(bool is_multiply, size_t rank, const size_t* out_shape, const size_t* a_strides,
                          const size_t* b_strides, size_t out_elements, const float* a, const float* b, float* out,
                          size_t start, size_t end) {
    end = std::min(end, out_elements);
    if (start >= end || rank == 0) return;

    const size_t inner_len = out_shape[rank - 1];
    const size_t sa = a_strides[rank - 1];
    const size_t sb = b_strides[rank - 1];
    size_t idx = start;

    while (idx < end) {
        const size_t inner_start = idx % inner_len;
        const size_t inner_end = std::min(inner_len, inner_start + (end - idx));
        size_t rem = idx / inner_len;
        size_t a_base = 0;
        size_t b_base = 0;
        for (size_t d = rank - 1; d-- > 0;) {
            const size_t coord = rem % out_shape[d];
            rem /= out_shape[d];
            a_base += coord * a_strides[d];
            b_base += coord * b_strides[d];
        }
        const size_t chunk = inner_end - inner_start;
        const float* ap = a + a_base + inner_start * sa;
        const float* bp = b + b_base + inner_start * sb;
        float* op = out + idx;

        if (sa == 1 && sb == 1) {
#if defined(__ARM_NEON)
            size_t i = 0;
            if (is_multiply) {
                for (; i + 4 <= chunk; i += 4) vst1q_f32(op + i, vmulq_f32(vld1q_f32(ap + i), vld1q_f32(bp + i)));
            } else {
                for (; i + 4 <= chunk; i += 4) vst1q_f32(op + i, vaddq_f32(vld1q_f32(ap + i), vld1q_f32(bp + i)));
            }
            for (; i < chunk; ++i) op[i] = is_multiply ? (ap[i] * bp[i]) : (ap[i] + bp[i]);
#else
            for (size_t i = 0; i < chunk; ++i) op[i] = is_multiply ? (ap[i] * bp[i]) : (ap[i] + bp[i]);
#endif
        } else if (sa == 1 && sb == 0) {
            const float bv = *bp;
#if defined(__ARM_NEON)
            const float32x4_t vb = vdupq_n_f32(bv);
            size_t i = 0;
            if (is_multiply) {
                for (; i + 4 <= chunk; i += 4) vst1q_f32(op + i, vmulq_f32(vld1q_f32(ap + i), vb));
            } else {
                for (; i + 4 <= chunk; i += 4) vst1q_f32(op + i, vaddq_f32(vld1q_f32(ap + i), vb));
            }
            for (; i < chunk; ++i) op[i] = is_multiply ? (ap[i] * bv) : (ap[i] + bv);
#else
            for (size_t i = 0; i < chunk; ++i) op[i] = is_multiply ? (ap[i] * bv) : (ap[i] + bv);
#endif
        } else if (sa == 0 && sb == 1) {
            const float av = *ap;
#if defined(__ARM_NEON)
            const float32x4_t va = vdupq_n_f32(av);
            size_t i = 0;
            if (is_multiply) {
                for (; i + 4 <= chunk; i += 4) vst1q_f32(op + i, vmulq_f32(va, vld1q_f32(bp + i)));
            } else {
                for (; i + 4 <= chunk; i += 4) vst1q_f32(op + i, vaddq_f32(va, vld1q_f32(bp + i)));
            }
            for (; i < chunk; ++i) op[i] = is_multiply ? (av * bp[i]) : (av + bp[i]);
#else
            for (size_t i = 0; i < chunk; ++i) op[i] = is_multiply ? (av * bp[i]) : (av + bp[i]);
#endif
        } else {
            for (size_t i = 0; i < chunk; ++i)
                op[i] = is_multiply ? (ap[i * sa] * bp[i * sb]) : (ap[i * sa] + bp[i * sb]);
        }
        idx += chunk;
    }
}

void transpose_elements(size_t elem_bytes, size_t rank, const size_t* in_shape, const size_t* out_strides_elems,
                        const uint8_t* src, uint8_t* dst, size_t outer_start, size_t outer_end) {
    if (rank == 0) {
        if (outer_start == 0 && outer_end > 0) std::memcpy(dst, src, elem_bytes);
        return;
    }
    const size_t inner_len = in_shape[rank - 1];
    size_t outer_total = 1;
    for (size_t d = 0; d + 1 < rank; ++d) outer_total *= in_shape[d];
    outer_end = std::min(outer_end, outer_total);
    const size_t dst_inner_stride = out_strides_elems[rank - 1];

    for (size_t o = outer_start; o < outer_end; ++o) {
        size_t rem = o;
        size_t dst_base = 0;
        for (size_t d = rank - 1; d-- > 0;) {
            const size_t coord = rem % in_shape[d];
            rem /= in_shape[d];
            dst_base += coord * out_strides_elems[d];
        }
        const uint8_t* sp = src + o * inner_len * elem_bytes;
        if (dst_inner_stride == 1) {
            std::memcpy(dst + dst_base * elem_bytes, sp, inner_len * elem_bytes);
        } else if (elem_bytes == 4) {
            const uint32_t* s32 = reinterpret_cast<const uint32_t*>(sp);
            uint32_t* d32 = reinterpret_cast<uint32_t*>(dst) + dst_base;
            for (size_t i = 0; i < inner_len; ++i) d32[i * dst_inner_stride] = s32[i];
        } else if (elem_bytes == 2) {
            const uint16_t* s16 = reinterpret_cast<const uint16_t*>(sp);
            uint16_t* d16 = reinterpret_cast<uint16_t*>(dst) + dst_base;
            for (size_t i = 0; i < inner_len; ++i) d16[i * dst_inner_stride] = s16[i];
        } else if (elem_bytes == 1) {
            uint8_t* d8 = dst + dst_base;
            for (size_t i = 0; i < inner_len; ++i) d8[i * dst_inner_stride] = sp[i];
        } else {
            for (size_t i = 0; i < inner_len; ++i) {
                std::memcpy(dst + (dst_base + i * dst_inner_stride) * elem_bytes, sp + i * elem_bytes, elem_bytes);
            }
        }
    }
}

} // namespace ynn
