// Licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/fp8.h"
#include "ynnpack/base/half.h"
#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/composites_internal.h"
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
    if (detail::arch_layer_norm_f32(rows, width, epsilon, input, residual_in, weight, bias, output_norm,
                                    output_residual, row_start, row_end)) {
        return;
    }
    const float inv_width = 1.0f / static_cast<float>(width);

    for (size_t r = row_start; r < row_end; ++r) {
        const float* x = input + r * width;
        const float* res = residual_in != nullptr ? residual_in + r * width : nullptr;
        float* y = output_norm + r * width;
        float* y_res = output_residual != nullptr ? output_residual + r * width : nullptr;

        float sum = 0.0f;
        for (size_t i = 0; i < width; ++i) {
            const float v = res != nullptr ? (x[i] + res[i]) : x[i];
            if (y_res != nullptr) y_res[i] = v;
            sum += v;
        }
        const float mean = sum * inv_width;

        float var_sum = 0.0f;
        const float* src = y_res != nullptr ? y_res : x;
        for (size_t j = 0; j < width; ++j) {
            const float v = (res != nullptr && y_res == nullptr) ? (x[j] + res[j]) : src[j];
            const float d = v - mean;
            var_sum += d * d;
        }
        const float inv_std = 1.0f / std::sqrt(var_sum * inv_width + epsilon);

        for (size_t k = 0; k < width; ++k) {
            const float v = (res != nullptr && y_res == nullptr) ? (x[k] + res[k]) : src[k];
            y[k] = (v - mean) * inv_std * weight[k] + bias[k];
        }
    }
}

void softmax_f32(size_t rows, size_t width, bool logarithmic, const float* input, float* output, size_t row_start,
                 size_t row_end) {
    row_end = std::min(row_end, rows);
    if (width == 0 || row_start >= row_end) return;
    if (detail::arch_softmax_f32(rows, width, logarithmic, input, output, row_start, row_end)) return;

    for (size_t r = row_start; r < row_end; ++r) {
        const float* x = input + r * width;
        float* y = output + r * width;

        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < width; ++i) max_val = std::max(max_val, x[i]);

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
            for (size_t j = 0; j < width; ++j) y[j] *= inv_sum;
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
                    YNN_DISABLE_FP_CONTRACT
                    for (size_t i = 0; i < n; ++i) {
                        const float v = x[i];
                        inner_buf[i] = (v + (v * v * v) * 0.044715f) * kSqrt2OverPi;
                    }
                }
                tanh_kernel(1, n, n * sizeof(float), inner_buf, n * sizeof(float), inner_buf, &tanh_params);
                {
                    YNN_DISABLE_FP_CONTRACT
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

        size_t i = detail::arch_broadcast_binary_inner_f32(is_multiply, chunk, sa, sb, ap, bp, op);
        if (sa == 1 && sb == 1) {
            for (; i < chunk; ++i) op[i] = is_multiply ? (ap[i] * bp[i]) : (ap[i] + bp[i]);
        } else if (sa == 1 && sb == 0) {
            const float bv = *bp;
            for (; i < chunk; ++i) op[i] = is_multiply ? (ap[i] * bv) : (ap[i] + bv);
        } else if (sa == 0 && sb == 1) {
            const float av = *ap;
            for (; i < chunk; ++i) op[i] = is_multiply ? (av * bp[i]) : (av + bp[i]);
        } else {
            for (; i < chunk; ++i) op[i] = is_multiply ? (ap[i * sa] * bp[i * sb]) : (ap[i * sa] + bp[i * sb]);
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

void slice_blocks(size_t outer, size_t src_stride_bytes, size_t dst_stride_bytes, size_t offset_bytes, const void* src,
                  void* dst, size_t begin, size_t end) {
    end = std::min(end, outer);
    const auto* s = static_cast<const uint8_t*>(src);
    auto* d = static_cast<uint8_t*>(dst);
    for (size_t b = begin; b < end; ++b) {
        std::memcpy(d + b * dst_stride_bytes, s + b * src_stride_bytes + offset_bytes, dst_stride_bytes);
    }
}

void gather_elements(size_t elem_bytes, size_t axis_len, size_t inner, size_t index_count, bool idx_is_i64,
                     const void* src, const void* indices, void* dst, size_t task_begin, size_t task_end) {
    const size_t slice_bytes = inner * elem_bytes;
    const auto* s = static_cast<const uint8_t*>(src);
    auto* d = static_cast<uint8_t*>(dst);
    const auto* idx64 = static_cast<const int64_t*>(indices);
    const auto* idx32 = static_cast<const int32_t*>(indices);
    for (size_t task = task_begin; task < task_end; ++task) {
        const size_t o = task / index_count;
        const size_t j = task % index_count;
        const size_t idx = static_cast<size_t>(idx_is_i64 ? idx64[j] : static_cast<int64_t>(idx32[j]));
        std::memcpy(d + task * slice_bytes, s + (o * axis_len + idx) * slice_bytes, slice_bytes);
    }
}

void scatter_rows(size_t batch_count, size_t dst_seq_len, size_t index_count, size_t row_bytes, const int32_t* indices,
                  const void* src, void* dst) {
    const auto* s = static_cast<const uint8_t*>(src);
    auto* d = static_cast<uint8_t*>(dst);
    for (size_t batch = 0; batch < batch_count; ++batch) {
        for (size_t i = 0; i < index_count; ++i) {
            std::memcpy(d + (batch * dst_seq_len + static_cast<size_t>(indices[i])) * row_bytes,
                        s + (batch * index_count + i) * row_bytes, row_bytes);
        }
    }
}

void broadcast_binary_i32(bool is_multiply, size_t rank, const size_t* out_shape, const size_t* a_strides,
                          const size_t* b_strides, size_t out_elements, const int32_t* a, const int32_t* b,
                          int32_t* out) {
    for (size_t idx = 0; idx < out_elements; ++idx) {
        size_t rem = idx, a_off = 0, b_off = 0;
        for (size_t d = rank; d-- > 0;) {
            const size_t coord = rem % out_shape[d];
            rem /= out_shape[d];
            a_off += coord * a_strides[d];
            b_off += coord * b_strides[d];
        }
        out[idx] = is_multiply ? (a[a_off] * b[b_off]) : (a[a_off] + b[b_off]);
    }
}

void broadcast_binary_i64(bool is_multiply, size_t rank, const size_t* out_shape, const size_t* a_strides,
                          const size_t* b_strides, size_t out_elements, const int64_t* a, const int64_t* b,
                          int64_t* out) {
    for (size_t idx = 0; idx < out_elements; ++idx) {
        size_t rem = idx, a_off = 0, b_off = 0;
        for (size_t d = rank; d-- > 0;) {
            const size_t coord = rem % out_shape[d];
            rem /= out_shape[d];
            a_off += coord * a_strides[d];
            b_off += coord * b_strides[d];
        }
        out[idx] = is_multiply ? (a[a_off] * b[b_off]) : (a[a_off] + b[b_off]);
    }
}

} // namespace ynn
