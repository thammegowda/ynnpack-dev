// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// The one-chunk online-softmax structure is adapted from llama.cpp's
// ggml_compute_forward_flash_attn_ext_f16_one_chunk. See
// LICENSES/llama.cpp.txt.

#include "ynnpack/composites/composites.h"
#include "ynnpack/base/thread_pool.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__APPLE__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#endif

namespace ynn {

void quantized_attention_prefill_f32(const quantized_attention_f32_params& params, const float* query,
                                     const int8_t* key, const int8_t* value, const float* mask, float* output,
                                     size_t row_start, size_t row_end, void* scratch);

namespace {

constexpr size_t QUERY_BLOCK_SIZE = 8;
constexpr size_t ATTENTION_PAGE_SIZE = 256;

size_t block_count(size_t width, const blockwise_quantization_f32_params& quantization) {
    if (quantization.block_size == 0) {
        return 0;
    }
    return (width + quantization.block_size - 1) / quantization.block_size;
}

bool valid_quantization(size_t width, const blockwise_quantization_f32_params& quantization) {
    return quantization.scales != nullptr && quantization.zero_points != nullptr &&
           quantization.num_blocks >= block_count(width, quantization);
}

int32_t dot_i8(size_t size, const int8_t* query, const int8_t* key, int32_t query_zero_point, int32_t key_zero_point) {
    int32_t result = 0;
    int32_t query_sum = 0;
    int32_t key_sum = 0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t dot = vdupq_n_s32(0);
    size_t channel = 0;
    for (; channel + 16 <= size; channel += 16) {
        const int8x16_t query_values = vld1q_s8(query + channel);
        const int8x16_t key_values = vld1q_s8(key + channel);
        dot = vdotq_s32(dot, query_values, key_values);
        if (key_zero_point != 0) {
            query_sum += vaddlvq_s8(query_values);
        }
        key_sum += vaddlvq_s8(key_values);
    }
    result = vaddvq_s32(dot);
    if (channel + 8 <= size) {
        const int8x8_t query_values = vld1_s8(query + channel);
        const int8x8_t key_values = vld1_s8(key + channel);
        result += vaddlvq_s16(vmull_s8(query_values, key_values));
        if (key_zero_point != 0) {
            query_sum += vaddlv_s8(query_values);
        }
        key_sum += vaddlv_s8(key_values);
        channel += 8;
    }
    for (; channel < size; ++channel) {
        result += query[channel] * key[channel];
        if (key_zero_point != 0) {
            query_sum += query[channel];
        }
        key_sum += key[channel];
    }
#else
    for (size_t channel = 0; channel < size; ++channel) {
        result += query[channel] * key[channel];
        if (key_zero_point != 0) {
            query_sum += query[channel];
        }
        key_sum += key[channel];
    }
#endif
    return result - query_zero_point * key_sum - key_zero_point * query_sum +
           static_cast<int32_t>(size) * query_zero_point * key_zero_point;
}

void quantize_query(size_t size, const float* input, int8_t* output, float& scale, int32_t& zero_point) {
    float minimum = 0.0f;
    float maximum = 0.0f;
    for (size_t channel = 0; channel < size; ++channel) {
        minimum = std::min(minimum, input[channel]);
        maximum = std::max(maximum, input[channel]);
    }
    const float range = maximum - minimum;
    scale = range == 0.0f ? 1.0f : range / 255.0f;
    zero_point =
        range == 0.0f ? 0 : static_cast<int32_t>(std::clamp(std::nearbyint(-minimum / scale), 0.0f, 255.0f)) - 128;
    for (size_t channel = 0; channel < size; ++channel) {
        const auto quantized = static_cast<int32_t>(
            std::clamp(std::nearbyint(input[channel] / scale) + static_cast<float>(zero_point), -128.0f, 127.0f));
        output[channel] = static_cast<int8_t>(quantized);
    }
}

void scale_output(size_t size, float scale, float* output) {
#if defined(__ARM_NEON)
    const float32x4_t scale_vector = vdupq_n_f32(scale);
    size_t channel = 0;
    for (; channel + 8 <= size; channel += 8) {
        vst1q_f32(output + channel, vmulq_f32(vld1q_f32(output + channel), scale_vector));
        vst1q_f32(output + channel + 4, vmulq_f32(vld1q_f32(output + channel + 4), scale_vector));
    }
    for (; channel < size; ++channel) {
        output[channel] *= scale;
    }
#else
    for (size_t channel = 0; channel < size; ++channel) {
        output[channel] *= scale;
    }
#endif
}

void accumulate_value(size_t size, const int8_t* value, int32_t zero_point, float scale, float weight, float* output) {
#if defined(__ARM_NEON)
    const int32x4_t zero = vdupq_n_s32(zero_point);
    const float32x4_t multiplier = vdupq_n_f32(scale * weight);
    size_t channel = 0;
    for (; channel + 8 <= size; channel += 8) {
        const int16x8_t value_i16 = vmovl_s8(vld1_s8(value + channel));
        const float32x4_t value0 = vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_low_s16(value_i16)), zero));
        const float32x4_t value1 = vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_high_s16(value_i16)), zero));
        vst1q_f32(output + channel, vfmaq_f32(vld1q_f32(output + channel), value0, multiplier));
        vst1q_f32(output + channel + 4, vfmaq_f32(vld1q_f32(output + channel + 4), value1, multiplier));
    }
    for (; channel < size; ++channel) {
        output[channel] += weight * (static_cast<float>(value[channel] - zero_point) * scale);
    }
#else
    for (size_t channel = 0; channel < size; ++channel) {
        output[channel] += weight * (static_cast<float>(value[channel] - zero_point) * scale);
    }
#endif
}

void compute_row(const quantized_attention_f32_params& params, const float* query, const int8_t* key,
                 const int8_t* value, const float* mask, size_t row, float* output, void* scratch) {
    uintptr_t cursor = (reinterpret_cast<uintptr_t>(scratch) + 63) & ~uintptr_t{63};
    auto* quantized_query = reinterpret_cast<int8_t*>(cursor);
    cursor += params.head_dim * sizeof(int8_t);
    cursor = (cursor + alignof(float) - 1) & ~uintptr_t{alignof(float) - 1};
    auto* query_scales = reinterpret_cast<float*>(cursor);
    const size_t query_blocks = (params.head_dim + QUERY_BLOCK_SIZE - 1) / QUERY_BLOCK_SIZE;
    cursor += query_blocks * sizeof(float);
    auto* query_zero_points = reinterpret_cast<int32_t*>(cursor);

    const size_t heads_per_group = params.query_heads / params.key_value_heads;
    const size_t key_sequence_stride = params.key_value_heads * params.head_dim;
    const size_t query_head = row % params.query_heads;
    const size_t query_row = row / params.query_heads;
    const size_t query_index = query_row % params.query_length;
    const size_t batch = query_row / params.query_length;
    const size_t key_value_head = query_head / heads_per_group;
    const size_t key_batch = params.key_batch_stride == 0 ? 0 : batch;
    const float* query_data = query + row * params.head_dim;
    std::fill(output, output + params.head_dim, 0.0f);

    for (size_t block = 0; block < query_blocks; ++block) {
        const size_t channel_start = block * QUERY_BLOCK_SIZE;
        const size_t channel_end = std::min(params.head_dim, channel_start + QUERY_BLOCK_SIZE);
        quantize_query(channel_end - channel_start, query_data + channel_start, quantized_query + channel_start,
                       query_scales[block], query_zero_points[block]);
    }

    const size_t key_base = key_batch * params.key_batch_stride + key_value_head * params.head_dim;
    const size_t mask_base = batch * params.mask_batch_stride + query_head * params.mask_head_stride +
                             query_index * params.mask_query_stride;
    float sum = 0.0f;
    float maximum = -std::numeric_limits<float>::infinity();
    const size_t padded_key_length =
        (params.key_length + ATTENTION_PAGE_SIZE - 1) / ATTENTION_PAGE_SIZE * ATTENTION_PAGE_SIZE;

    for (size_t page_start = 0; page_start < padded_key_length; page_start += ATTENTION_PAGE_SIZE) {
        const size_t page_end = std::min(page_start + ATTENTION_PAGE_SIZE, padded_key_length);
        for (size_t token = page_start; token < page_end; ++token) {
            if (token >= params.key_length) {
                continue;
            }
            const float mask_value = mask == nullptr ? 0.0f : mask[mask_base + token * params.mask_key_stride];
            if (mask_value == -std::numeric_limits<float>::infinity()) {
                continue;
            }

            const size_t key_offset = key_base + (params.key_start + token) * key_sequence_stride;
            float score = 0.0f;
            for (size_t query_block = 0; query_block < query_blocks; ++query_block) {
                const size_t query_start = query_block * QUERY_BLOCK_SIZE;
                const size_t query_end = std::min(params.head_dim, query_start + QUERY_BLOCK_SIZE);
                for (size_t channel_start = query_start; channel_start < query_end;) {
                    const size_t key_block = channel_start / params.key_quantization.block_size;
                    const size_t channel_end =
                        std::min(query_end, (key_block + 1) * params.key_quantization.block_size);
                    const int32_t dot = dot_i8(channel_end - channel_start, quantized_query + channel_start,
                                               key + key_offset + channel_start, query_zero_points[query_block],
                                               params.key_quantization.zero_points[key_block]);
                    score +=
                        static_cast<float>(dot) * query_scales[query_block] * params.key_quantization.scales[key_block];
                    channel_start = channel_end;
                }
            }
            score = score * params.scale + mask_value;

            const float old_maximum = maximum;
            float output_scale = 1.0f;
            float weight = 1.0f;
            if (score > maximum) {
                maximum = score;
                output_scale = std::exp(old_maximum - maximum);
                scale_output(params.head_dim, output_scale, output);
            } else {
                weight = std::exp(score - maximum);
            }

            const size_t value_offset = key_base + (params.key_start + token) * key_sequence_stride;
            for (size_t block = 0; block < block_count(params.head_dim, params.value_quantization); ++block) {
                const size_t channel_start = block * params.value_quantization.block_size;
                const size_t channel_end =
                    std::min(params.head_dim, channel_start + params.value_quantization.block_size);
                accumulate_value(channel_end - channel_start, value + value_offset + channel_start,
                                 params.value_quantization.zero_points[block], params.value_quantization.scales[block],
                                 weight, output + channel_start);
            }
            sum = sum * output_scale + weight;
        }
    }
    scale_output(params.head_dim, sum == 0.0f ? 0.0f : 1.0f / sum, output);
}

} // namespace

size_t quantized_attention_f32_scratch_size(const quantized_attention_f32_params& params) {
    if (params.query_length > 1) {
        if (params.key_value_heads == 0 || params.query_heads % params.key_value_heads != 0) {
            return 0;
        }
        const size_t heads_per_group = params.query_heads / params.key_value_heads;
        return (params.key_length + params.head_dim) * heads_per_group * sizeof(float) + 64;
    }
    const size_t query_blocks = (params.head_dim + QUERY_BLOCK_SIZE - 1) / QUERY_BLOCK_SIZE;
    return params.head_dim * sizeof(int8_t) + query_blocks * (sizeof(float) + sizeof(int32_t)) + 128;
}

size_t quantized_attention_f32_work_items(const quantized_attention_f32_params& params) {
    const size_t heads = params.query_length > 1 ? params.key_value_heads : params.query_heads;
    return params.batch_size * params.query_length * heads;
}

void quantized_attention_f32(const quantized_attention_f32_params& params, const float* query, const int8_t* key,
                             const int8_t* value, const float* mask, float* output, size_t row_start, size_t row_end,
                             void* scratch) {
    if (params.query_length > 1) {
        quantized_attention_prefill_f32(params, query, key, value, mask, output, row_start, row_end, scratch);
        return;
    }
    if (scratch == nullptr || !valid_quantization(params.head_dim, params.key_quantization) ||
        !valid_quantization(params.head_dim, params.value_quantization) || params.query_heads == 0 ||
        params.key_value_heads == 0 || params.query_heads % params.key_value_heads != 0) {
        return;
    }
    const size_t rows = quantized_attention_f32_work_items(params);
    row_end = std::min(row_end, rows);
    for (size_t row = row_start; row < row_end; ++row) {
        float* output_data = output + row * params.head_dim;
        compute_row(params, query, key, value, mask, row, output_data, scratch);
    }
}

namespace {

float dot_f32_lanes8(const float* left, const float* right, size_t count) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    size_t index = 0;
#if defined(__ARM_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (; index + 8 <= count; index += 8) {
        acc0 = vaddq_f32(acc0, vmulq_f32(vld1q_f32(left + index), vld1q_f32(right + index)));
        acc1 = vaddq_f32(acc1, vmulq_f32(vld1q_f32(left + index + 4), vld1q_f32(right + index + 4)));
    }
    float lanes[8];
    vst1q_f32(lanes, acc0);
    vst1q_f32(lanes + 4, acc1);
#else
    float lanes[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (; index + 8 <= count; index += 8) {
        for (size_t lane = 0; lane < 8; ++lane) {
            lanes[lane] += left[index + lane] * right[index + lane];
        }
    }
#endif
    float sum = 0.0f;
    for (size_t lane = 0; lane < 8; ++lane) sum += lanes[lane];
    for (; index < count; ++index) sum += left[index] * right[index];
    return sum;
}

inline void accumulate_weighted_f32(float* dst, const float* src, float weight, size_t count) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    size_t index = 0;
#if defined(__ARM_NEON)
    const float32x4_t vw = vdupq_n_f32(weight);
    for (; index + 8 <= count; index += 8) {
        vst1q_f32(dst + index, vaddq_f32(vld1q_f32(dst + index), vmulq_f32(vld1q_f32(src + index), vw)));
        vst1q_f32(dst + index + 4, vaddq_f32(vld1q_f32(dst + index + 4), vmulq_f32(vld1q_f32(src + index + 4), vw)));
    }
#endif
    for (; index < count; ++index) dst[index] += weight * src[index];
}

bool is_single_gqa8_step(const attention_f32_params& params) {
    return params.batch_size == 1 && params.query_length == 1 && params.key_value_heads == 1 &&
           params.query_heads == 8 && params.mask_head_stride == 0;
}

} // namespace

size_t attention_f32_work_items(const attention_f32_params& params) {
    if (is_single_gqa8_step(params)) return 4;
    return params.batch_size * params.query_length * params.query_heads;
}

size_t attention_f32_scratch_size(const attention_f32_params& params) {
    const size_t keys = params.key_length ? params.key_length : params.total_keys;
    return (is_single_gqa8_step(params) ? 2 : 1) * keys;
}

void attention_f32(const attention_f32_params& params, const float* query, const float* key_data,
                   const float* value_data, const float* mask, float* output_data, size_t item_start, size_t item_end,
                   float* scratch) {
    if (params.query_heads == 0 || params.key_value_heads == 0 || params.query_heads % params.key_value_heads != 0) {
        return;
    }
    const size_t rows = params.query_length;
    const size_t heads = params.query_heads;
    const size_t key_heads = params.key_value_heads;
    const size_t head_width = params.head_dim;
    const size_t width = heads * head_width;
    const size_t kv_width = key_heads * head_width;
    const size_t group_size = heads / key_heads;
    const size_t keys = params.key_length ? params.key_length : params.total_keys;

    if (is_single_gqa8_step(params)) {
        item_end = std::min<size_t>(item_end, 4);
        if (item_start >= item_end) return;
        const float* k_base = key_data + params.key_start * head_width;
        const float* v_base = value_data + params.key_start * head_width;
        size_t active_keys = keys;
        if (mask != nullptr) {
            while (active_keys > 0 && mask[(active_keys - 1) * params.mask_key_stride] <= -1e8f) --active_keys;
        }
        for (size_t item = item_start; item < item_end; ++item) {
            const size_t h_begin = item * 2;
            if (active_keys == 0) {
                std::fill_n(output_data + h_begin * head_width, 2 * head_width, 0.0f);
                continue;
            }
            float largest0 = -std::numeric_limits<float>::infinity();
            float largest1 = -std::numeric_limits<float>::infinity();
            float* scores0 = scratch;
            float* scores1 = scratch + active_keys;
            const float* q0 = query + h_begin * head_width;
            const float* q1 = q0 + head_width;
            for (size_t k_idx = 0; k_idx < active_keys; ++k_idx) {
                const float b = mask != nullptr ? mask[k_idx * params.mask_key_stride] : 0.0f;
                if (b <= -1e8f) {
                    scores0[k_idx] = -std::numeric_limits<float>::infinity();
                    scores1[k_idx] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                const float* k_row = k_base + k_idx * head_width;
                const float s0 = dot_f32_lanes8(q0, k_row, head_width) * params.scale + b;
                const float s1 = dot_f32_lanes8(q1, k_row, head_width) * params.scale + b;
                scores0[k_idx] = s0;
                scores1[k_idx] = s1;
                largest0 = std::max(largest0, s0);
                largest1 = std::max(largest1, s1);
            }
            float total0 = 0.0f;
            float total1 = 0.0f;
            for (size_t k_idx = 0; k_idx < active_keys; ++k_idx) {
                scores0[k_idx] = scores0[k_idx] == -std::numeric_limits<float>::infinity()
                                     ? 0.0f
                                     : std::exp(scores0[k_idx] - largest0);
                scores1[k_idx] = scores1[k_idx] == -std::numeric_limits<float>::infinity()
                                     ? 0.0f
                                     : std::exp(scores1[k_idx] - largest1);
                total0 += scores0[k_idx];
                total1 += scores1[k_idx];
            }
            const float inv_total0 = 1.0f / total0;
            const float inv_total1 = 1.0f / total1;
            float* out0 = output_data + h_begin * head_width;
            float* out1 = out0 + head_width;
            std::fill_n(out0, 2 * head_width, 0.0f);
            for (size_t k_idx = 0; k_idx < active_keys; ++k_idx) {
                const float b = mask != nullptr ? mask[k_idx * params.mask_key_stride] : 0.0f;
                if (b <= -1e8f) continue;
                const float* v_row = v_base + k_idx * head_width;
                accumulate_weighted_f32(out0, v_row, scores0[k_idx] * inv_total0, head_width);
                accumulate_weighted_f32(out1, v_row, scores1[k_idx] * inv_total1, head_width);
            }
        }
        return;
    }

    const size_t items = params.batch_size * rows * heads;
    item_end = std::min(item_end, items);
    for (size_t item = item_start; item < item_end; ++item) {
        const size_t row = item % rows;
        const size_t head = (item / rows) % heads;
        const size_t sample = item / (rows * heads);
        const size_t kv_head = head / group_size;
        const float* q = query + (sample * rows + row) * width + head * head_width;
        const float* k =
            key_data + sample * params.key_batch_stride + (params.key_start * key_heads + kv_head) * head_width;
        const float* v =
            value_data + sample * params.key_batch_stride + (params.key_start * key_heads + kv_head) * head_width;
        const float* bias = mask != nullptr ? mask + sample * params.mask_batch_stride +
                                                  head * params.mask_head_stride + row * params.mask_query_stride
                                            : nullptr;
        size_t active_keys = keys;
        if (bias != nullptr) {
            while (active_keys > 0 && bias[(active_keys - 1) * params.mask_key_stride] <= -1e8f) --active_keys;
        }
        if (active_keys == 0) {
            std::fill_n(output_data + (sample * rows + row) * width + head * head_width, head_width, 0.0f);
            continue;
        }
        float largest = -std::numeric_limits<float>::infinity();
        for (size_t k_idx = 0; k_idx < active_keys; ++k_idx) {
            const float b = bias != nullptr ? bias[k_idx * params.mask_key_stride] : 0.0f;
            if (b <= -1e8f) {
                scratch[k_idx] = -std::numeric_limits<float>::infinity();
                continue;
            }
            scratch[k_idx] = dot_f32_lanes8(q, k + k_idx * kv_width, head_width) * params.scale + b;
            largest = std::max(largest, scratch[k_idx]);
        }
        float total = 0.0f;
        for (size_t k_idx = 0; k_idx < active_keys; ++k_idx) {
            scratch[k_idx] =
                scratch[k_idx] == -std::numeric_limits<float>::infinity() ? 0.0f : std::exp(scratch[k_idx] - largest);
            total += scratch[k_idx];
        }
        const float inv_total = 1.0f / total;
        float* out = output_data + (sample * rows + row) * width + head * head_width;
        std::fill_n(out, head_width, 0.0f);
        for (size_t k_idx = 0; k_idx < active_keys; ++k_idx) {
            if (scratch[k_idx] == 0.0f) continue;
            const float weight = scratch[k_idx] * inv_total;
            accumulate_weighted_f32(out, v + k_idx * kv_width, weight, head_width);
        }
    }
}

void attention_prefill_f32(const attention_f32_params& params, const float* query_data, const float* key_data,
                           const float* value_data, const float* mask, float* output_data) {
    if (params.query_heads == 0 || params.key_value_heads == 0 || params.query_heads % params.key_value_heads != 0) {
        return;
    }
    const size_t batch = params.batch_size;
    const size_t rows = params.query_length;
    const size_t heads = params.query_heads;
    const size_t key_heads = params.key_value_heads;
    const size_t head_width = params.head_dim;
    const size_t width = heads * head_width;
    const size_t kv_width = key_heads * head_width;
    const size_t keys = params.key_length ? params.key_length : params.total_keys;
    const size_t group_size = heads / key_heads;
    const size_t q_group_rows = group_size * rows;
    const size_t kv_groups = batch * key_heads;
    if (rows == 0 || keys == 0 || head_width == 0) return;

    struct PrefillScratch {
        std::vector<float> q_buf;
        std::vector<float> k_buf;
        std::vector<float> v_buf;
        std::vector<float> scores;
        std::vector<float> out_buf;
    };
    thread_local PrefillScratch tls;
    tls.q_buf.resize(q_group_rows * head_width);
    if (key_heads > 1) {
        tls.k_buf.resize(keys * head_width);
        tls.v_buf.resize(keys * head_width);
    }
    tls.scores.resize(q_group_rows * keys);
    tls.out_buf.resize(q_group_rows * head_width);
    auto& q_buf = tls.q_buf;
    auto& k_buf = tls.k_buf;
    auto& v_buf = tls.v_buf;
    auto& scores = tls.scores;
    auto& out_buf = tls.out_buf;

    for (size_t g = 0; g < kv_groups; ++g) {
        const size_t sample = g / key_heads;
        const size_t kv_head = g % key_heads;
        const size_t first_q_head = kv_head * group_size;

        const float* q_src = query_data + sample * rows * width + first_q_head * head_width;
        for (size_t gh = 0; gh < group_size; ++gh) {
            for (size_t r = 0; r < rows; ++r) {
                std::memcpy(q_buf.data() + (gh * rows + r) * head_width, q_src + r * width + gh * head_width,
                            head_width * sizeof(float));
            }
        }

        const float* k_src =
            key_data + sample * params.key_batch_stride + (params.key_start * key_heads + kv_head) * head_width;
        const float* v_src =
            value_data + sample * params.key_batch_stride + (params.key_start * key_heads + kv_head) * head_width;
        const float* k_mat = k_src;
        const float* v_mat = v_src;
        if (key_heads > 1) {
            for (size_t t = 0; t < keys; ++t) {
                std::memcpy(k_buf.data() + t * head_width, k_src + t * kv_width, head_width * sizeof(float));
                std::memcpy(v_buf.data() + t * head_width, v_src + t * kv_width, head_width * sizeof(float));
            }
            k_mat = k_buf.data();
            v_mat = v_buf.data();
        }

#if defined(__APPLE__)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(q_group_rows), static_cast<int>(keys),
                    static_cast<int>(head_width), params.scale, q_buf.data(), static_cast<int>(head_width), k_mat,
                    static_cast<int>(head_width), 0.0f, scores.data(), static_cast<int>(keys));
#else
        dense_linear_plan qk_plan{};
        if (make_dense_linear_plan(ynn_type_fp32, ynn_type_fp32, ynn_type_fp32, keys, head_width, &qk_plan) !=
            ynn_status_success) {
            return;
        }
        aligned_byte_buffer packed_k(qk_plan.total_bytes);
        pack_dense_linear_tiles(qk_plan, true, k_mat, head_width * sizeof(float), 0, qk_plan.blocks_n, packed_k.data());
        bool ta = false;
        size_t tm = 1, ab = 0;
        dense_linear_select_kernel(qk_plan, q_group_rows, &ta, &tm, &ab);
        aligned_byte_buffer pa(ab);
        const void* eff_q = q_buf.data();
        if (ta) {
            dense_linear_pack_a(qk_plan, q_group_rows, tm, eff_q, head_width * sizeof(float), pa.data());
            eff_q = pa.data();
        }
        dense_linear_run_tiles(qk_plan, q_group_rows, ta, tm, eff_q, head_width * sizeof(float), packed_k.data(),
                               nullptr, scores.data(), keys, 0, qk_plan.blocks_n);
        if (params.scale != 1.0f) {
            for (float& s : scores) s *= params.scale;
        }
#endif

        const auto mask_and_softmax = [&](size_t row_begin, size_t row_end) {
            for (size_t row_idx = row_begin; row_idx < row_end; ++row_idx) {
                const size_t gh = row_idx / rows;
                const size_t r = row_idx % rows;
                const size_t head = first_q_head + gh;
                float* s_row = scores.data() + row_idx * keys;
                if (mask != nullptr) {
                    const float* m_row = mask + sample * params.mask_batch_stride + head * params.mask_head_stride +
                                         r * params.mask_query_stride;
                    if (params.mask_key_stride == 1) {
                        for (size_t t = 0; t < keys; ++t) s_row[t] += m_row[t];
                    } else if (params.mask_key_stride == 0) {
                        const float mv = m_row[0];
                        for (size_t t = 0; t < keys; ++t) s_row[t] += mv;
                    } else {
                        for (size_t t = 0; t < keys; ++t) s_row[t] += m_row[t * params.mask_key_stride];
                    }
                }
                bool any_valid = false;
                for (size_t t = 0; t < keys; ++t) {
                    if (s_row[t] > -1e8f) {
                        any_valid = true;
                        break;
                    }
                }
                if (!any_valid) {
                    std::fill_n(s_row, keys, 0.0f);
                } else {
                    softmax_f32(1, keys, false, s_row, s_row, 0, 1);
                }
            }
        };
        if (q_group_rows * keys < 2048 || thread_count() <= 1) {
            mask_and_softmax(0, q_group_rows);
        } else {
            struct SoftmaxCtx {
                decltype(mask_and_softmax)* fn;
            } sctx{&mask_and_softmax};
            parallel_ranges(q_group_rows, 1, &sctx, [](void* raw, size_t begin, size_t end) {
                (*static_cast<SoftmaxCtx*>(raw)->fn)(begin, end);
            });
        }

#if defined(__APPLE__)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(q_group_rows),
                    static_cast<int>(head_width), static_cast<int>(keys), 1.0f, scores.data(), static_cast<int>(keys),
                    v_mat, static_cast<int>(head_width), 0.0f, out_buf.data(), static_cast<int>(head_width));
#else
        dense_linear_plan pv_plan{};
        if (make_dense_linear_plan(ynn_type_fp32, ynn_type_fp32, ynn_type_fp32, head_width, keys, &pv_plan) !=
            ynn_status_success) {
            return;
        }
        aligned_byte_buffer packed_v(pv_plan.total_bytes);
        pack_dense_linear_tiles(pv_plan, false, v_mat, head_width * sizeof(float), 0, pv_plan.blocks_n,
                                packed_v.data());
        bool ta2 = false;
        size_t tm2 = 1, ab2 = 0;
        dense_linear_select_kernel(pv_plan, q_group_rows, &ta2, &tm2, &ab2);
        aligned_byte_buffer pa2(ab2);
        const void* eff_p = scores.data();
        if (ta2) {
            dense_linear_pack_a(pv_plan, q_group_rows, tm2, eff_p, keys * sizeof(float), pa2.data());
            eff_p = pa2.data();
        }
        dense_linear_run_tiles(pv_plan, q_group_rows, ta2, tm2, eff_p, keys * sizeof(float), packed_v.data(), nullptr,
                               out_buf.data(), head_width, 0, pv_plan.blocks_n);
#endif

        float* out_dst = output_data + sample * rows * width + first_q_head * head_width;
        for (size_t gh = 0; gh < group_size; ++gh) {
            for (size_t r = 0; r < rows; ++r) {
                std::memcpy(out_dst + r * width + gh * head_width, out_buf.data() + (gh * rows + r) * head_width,
                            head_width * sizeof(float));
            }
        }
    }
}

} // namespace ynn
