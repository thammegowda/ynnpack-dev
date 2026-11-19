// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/composites_internal.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/unary/unary.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ynn {
namespace {

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

void accumulate_grouped_value(size_t size, const int8_t* value, int32_t zero_point, float scale,
                              const float* probabilities, size_t probability_stride, size_t heads, float* output,
                              size_t output_stride) {
    if (detail::arch_accumulate_grouped_value(size, value, zero_point, scale, probabilities, probability_stride, heads,
                                              output, output_stride)) {
        return;
    }
    for (size_t channel = 0; channel < size; ++channel) {
        const float dequantized = static_cast<float>(value[channel] - zero_point) * scale;
        for (size_t head = 0; head < heads; ++head) {
            output[head * output_stride + channel] += probabilities[head * probability_stride] * dequantized;
        }
    }
}

} // namespace

void quantized_attention_prefill_f32(const quantized_attention_f32_params& params, const float* query,
                                     const int8_t* key, const int8_t* value, const float* mask, float* output,
                                     size_t row_start, size_t row_end, void* scratch) {
    if (params.query_length <= 1 || scratch == nullptr ||
        !valid_quantization(params.head_dim, params.key_quantization) ||
        !valid_quantization(params.head_dim, params.value_quantization) || params.query_heads == 0 ||
        params.key_value_heads == 0 || params.query_heads % params.key_value_heads != 0) {
        return;
    }
    const auto exp_kernel = get_unary_kernel(ynn_unary_exp, ynn_type_fp32, ynn_type_fp32);
    const auto sum_kernel = get_sum_kernel(ynn_type_fp32, ynn_type_fp32).k1;
    if (exp_kernel == nullptr || sum_kernel == nullptr) {
        return;
    }
    const auto exp_params = get_unary_params(ynn_unary_exp);
    auto* score_data = reinterpret_cast<float*>((reinterpret_cast<uintptr_t>(scratch) + 63) & ~uintptr_t{63});
    const size_t heads_per_group = params.query_heads / params.key_value_heads;
    const size_t key_sequence_stride = params.key_value_heads * params.head_dim;
    float* query_tile = score_data + heads_per_group * params.key_length;
    const auto normalize = [&](float* scores, size_t active_keys) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (size_t token = 0; token < active_keys; ++token) {
            maximum = std::max(maximum, scores[token]);
        }
        for (size_t token = 0; token < active_keys; ++token) {
            scores[token] -= maximum;
        }
        exp_kernel(1, active_keys, active_keys * sizeof(float), scores, active_keys * sizeof(float), scores,
                   &exp_params);
        float exponential_sum = 0.0f;
        sum_kernel(1, active_keys, active_keys * sizeof(float), scores, &exponential_sum, nullptr);
        const float inverse_sum = exponential_sum == 0.0f ? 0.0f : 1.0f / exponential_sum;
        for (size_t token = 0; token < active_keys; ++token) {
            scores[token] *= inverse_sum;
        }
    };

    constexpr size_t MAX_HEADS_PER_GROUP = 8;
    const size_t work_items = quantized_attention_f32_work_items(params);
    row_end = std::min(row_end, work_items);
    for (size_t work = row_start; work < row_end; ++work) {
        const size_t key_value_head = work % params.key_value_heads;
        const size_t query_row = work / params.key_value_heads;
        const size_t query_index = query_row % params.query_length;
        const size_t batch = query_row / params.query_length;
        const size_t key_batch = params.key_batch_stride == 0 ? 0 : batch;
        const size_t first_query_head = key_value_head * heads_per_group;
        const size_t key_base = key_batch * params.key_batch_stride + key_value_head * params.head_dim;

        size_t active_keys = params.key_length;
        if (mask != nullptr && params.mask_head_stride == 0) {
            const float* row_mask = mask + batch * params.mask_batch_stride + query_index * params.mask_query_stride;
            while (active_keys > 0 && row_mask[(active_keys - 1) * params.mask_key_stride] <= -1e8f) {
                --active_keys;
            }
        }
        float* output_data = output + (query_row * params.query_heads + first_query_head) * params.head_dim;
        if (active_keys == 0) {
            std::fill(output_data, output_data + heads_per_group * params.head_dim, 0.0f);
            continue;
        }

        for (size_t channel = 0; channel < params.head_dim; ++channel) {
            for (size_t group_head = 0; group_head < heads_per_group; ++group_head) {
                query_tile[channel * heads_per_group + group_head] =
                    query[(query_row * params.query_heads + first_query_head + group_head) * params.head_dim + channel];
            }
        }

        for (size_t token = 0; token < active_keys; ++token) {
            const size_t key_offset = key_base + (params.key_start + token) * key_sequence_stride;
            float scores[MAX_HEADS_PER_GROUP]{};
            if (!detail::arch_score_grouped_key_token(params.head_dim, heads_per_group, query_tile, key + key_offset,
                                                      params.key_quantization, scores)) {
                for (size_t block = 0; block < block_count(params.head_dim, params.key_quantization); ++block) {
                    const size_t channel_start = block * params.key_quantization.block_size;
                    const size_t channel_end =
                        std::min(params.head_dim, channel_start + params.key_quantization.block_size);
                    const float key_scale = params.key_quantization.scales[block];
                    const int32_t key_zero_point = params.key_quantization.zero_points[block];
                    for (size_t channel = channel_start; channel < channel_end; ++channel) {
                        const float dequantized =
                            static_cast<float>(key[key_offset + channel] - key_zero_point) * key_scale;
                        for (size_t head = 0; head < heads_per_group; ++head) {
                            scores[head] =
                                std::fma(query_tile[channel * heads_per_group + head], dequantized, scores[head]);
                        }
                    }
                }
            }
            for (size_t group_head = 0; group_head < heads_per_group; ++group_head) {
                const size_t query_head = first_query_head + group_head;
                float score = scores[group_head] * params.scale;
                if (mask != nullptr) {
                    score += mask[batch * params.mask_batch_stride + query_head * params.mask_head_stride +
                                  query_index * params.mask_query_stride + token * params.mask_key_stride];
                }
                score_data[group_head * active_keys + token] = score;
            }
        }

        for (size_t group_head = 0; group_head < heads_per_group; ++group_head) {
            normalize(score_data + group_head * active_keys, active_keys);
        }

        std::fill(output_data, output_data + heads_per_group * params.head_dim, 0.0f);
        for (size_t token = 0; token < active_keys; ++token) {
            const size_t value_offset = key_base + (params.key_start + token) * key_sequence_stride;
            for (size_t block = 0; block < block_count(params.head_dim, params.value_quantization); ++block) {
                const size_t channel_start = block * params.value_quantization.block_size;
                const size_t channel_end =
                    std::min(params.head_dim, channel_start + params.value_quantization.block_size);
                accumulate_grouped_value(channel_end - channel_start, value + value_offset + channel_start,
                                         params.value_quantization.zero_points[block],
                                         params.value_quantization.scales[block], score_data + token, active_keys,
                                         heads_per_group, output_data + channel_start, params.head_dim);
            }
        }
    }
}

} // namespace ynn