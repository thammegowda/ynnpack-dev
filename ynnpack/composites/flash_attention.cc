// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// The one-chunk online-softmax structure is adapted from llama.cpp's
// ggml_compute_forward_flash_attn_ext_f16_one_chunk. See
// LICENSES/llama.cpp.txt.

#include "ynnpack/composites/composites.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__ARM_NEON)
#include <arm_neon.h>
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
                    score += static_cast<float>(dot) * query_scales[query_block] *
                             params.key_quantization.scales[key_block];
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
                                 params.value_quantization.zero_points[block],
                                 params.value_quantization.scales[block], weight, output + channel_start);
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

} // namespace ynn
