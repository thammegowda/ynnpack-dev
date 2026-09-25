// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/composites/composites.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/unary/unary.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

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
#if defined(__ARM_NEON)
    const int32x4_t zero = vdupq_n_s32(zero_point);
    const float32x4_t scale_vector = vdupq_n_f32(scale);
    size_t channel = 0;
    for (; channel + 8 <= size; channel += 8) {
        const int16x8_t value_i16 = vmovl_s8(vld1_s8(value + channel));
        const float32x4_t value0 =
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_low_s16(value_i16)), zero)), scale_vector);
        const float32x4_t value1 =
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_high_s16(value_i16)), zero)), scale_vector);
        for (size_t head = 0; head < heads; ++head) {
            const float32x4_t weight = vdupq_n_f32(probabilities[head * probability_stride]);
            float* head_output = output + head * output_stride + channel;
            vst1q_f32(head_output, vfmaq_f32(vld1q_f32(head_output), value0, weight));
            vst1q_f32(head_output + 4, vfmaq_f32(vld1q_f32(head_output + 4), value1, weight));
        }
    }
    for (; channel < size; ++channel) {
        const float dequantized = static_cast<float>(value[channel] - zero_point) * scale;
        for (size_t head = 0; head < heads; ++head) {
            output[head * output_stride + channel] += probabilities[head * probability_stride] * dequantized;
        }
    }
#else
    for (size_t channel = 0; channel < size; ++channel) {
        const float dequantized = static_cast<float>(value[channel] - zero_point) * scale;
        for (size_t head = 0; head < heads; ++head) {
            output[head * output_stride + channel] += probabilities[head * probability_stride] * dequantized;
        }
    }
#endif
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
    const auto normalize = [&](float* scores) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (size_t token = 0; token < params.key_length; ++token) {
            maximum = std::max(maximum, scores[token]);
        }
        for (size_t token = 0; token < params.key_length; ++token) {
            scores[token] -= maximum;
        }
        exp_kernel(1, params.key_length, params.key_length * sizeof(float), scores, params.key_length * sizeof(float),
                   scores, &exp_params);
        float exponential_sum = 0.0f;
        sum_kernel(1, params.key_length, params.key_length * sizeof(float), scores, &exponential_sum, nullptr);
        const float inverse_sum = exponential_sum == 0.0f ? 0.0f : 1.0f / exponential_sum;
        for (size_t token = 0; token < params.key_length; ++token) {
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

        for (size_t channel = 0; channel < params.head_dim; ++channel) {
            for (size_t group_head = 0; group_head < heads_per_group; ++group_head) {
                query_tile[channel * heads_per_group + group_head] =
                    query[(query_row * params.query_heads + first_query_head + group_head) * params.head_dim + channel];
            }
        }

        for (size_t token = 0; token < params.key_length; ++token) {
            const size_t key_offset = key_base + (params.key_start + token) * key_sequence_stride;
            float scores[MAX_HEADS_PER_GROUP]{};
#if defined(__ARM_NEON)
            float32x4_t vector_scores[MAX_HEADS_PER_GROUP / 4]{};
#endif
            for (size_t block = 0; block < block_count(params.head_dim, params.key_quantization); ++block) {
                const size_t channel_start = block * params.key_quantization.block_size;
                const size_t channel_end =
                    std::min(params.head_dim, channel_start + params.key_quantization.block_size);
                const float key_scale = params.key_quantization.scales[block];
                const int32_t key_zero_point = params.key_quantization.zero_points[block];
                for (size_t channel = channel_start; channel < channel_end; ++channel) {
                    const float dequantized =
                        static_cast<float>(key[key_offset + channel] - key_zero_point) * key_scale;
#if defined(__ARM_NEON)
                    size_t head = 0;
                    for (; head + 4 <= heads_per_group; head += 4) {
                        vector_scores[head / 4] =
                            vfmaq_n_f32(vector_scores[head / 4],
                                        vld1q_f32(query_tile + channel * heads_per_group + head), dequantized);
                    }
                    for (; head < heads_per_group; ++head) {
#else
                    for (size_t head = 0; head < heads_per_group; ++head) {
#endif
                        scores[head] =
                            std::fma(query_tile[channel * heads_per_group + head], dequantized, scores[head]);
                    }
                }
            }
#if defined(__ARM_NEON)
            for (size_t head = 0; head + 4 <= heads_per_group; head += 4) {
                vst1q_f32(scores + head, vector_scores[head / 4]);
            }
#endif
            for (size_t group_head = 0; group_head < heads_per_group; ++group_head) {
                const size_t query_head = first_query_head + group_head;
                float score = scores[group_head] * params.scale;
                if (mask != nullptr) {
                    score += mask[batch * params.mask_batch_stride + query_head * params.mask_head_stride +
                                  query_index * params.mask_query_stride + token * params.mask_key_stride];
                }
                score_data[group_head * params.key_length + token] = score;
            }
        }

        for (size_t group_head = 0; group_head < heads_per_group; ++group_head) {
            normalize(score_data + group_head * params.key_length);
        }

        float* output_data = output + (query_row * params.query_heads + first_query_head) * params.head_dim;
        std::fill(output_data, output_data + heads_per_group * params.head_dim, 0.0f);
        for (size_t token = 0; token < params.key_length; ++token) {
            const size_t value_offset = key_base + (params.key_start + token) * key_sequence_stride;
            for (size_t block = 0; block < block_count(params.head_dim, params.value_quantization); ++block) {
                const size_t channel_start = block * params.value_quantization.block_size;
                const size_t channel_end =
                    std::min(params.head_dim, channel_start + params.value_quantization.block_size);
                accumulate_grouped_value(channel_end - channel_start, value + value_offset + channel_start,
                                         params.value_quantization.zero_points[block],
                                         params.value_quantization.scales[block], score_data + token, params.key_length,
                                         heads_per_group, output_data + channel_start, params.head_dim);
            }
        }
    }
}

} // namespace ynn