// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/composites/composites.h"

namespace ynn {
namespace {

TEST(LlmCompositesTest, ArgmaxAndSymmetricInt8Quantization) {
  const std::vector<float> logits = {
      -1.0f, 3.5f,
      2.0f,  3.5f,
      1.0f,  std::numeric_limits<float>::quiet_NaN(),
      5.0f,  2.0f,
      0.0f,  std::numeric_limits<float>::infinity(),
      1.0f,  2.0f,
  };
  std::vector<int32_t> tokens(3, -99);
  argmax_rows_f32(3, 4, logits.data(), tokens.data());
  EXPECT_EQ(tokens[0], 1);
  EXPECT_EQ(tokens[1], -1);
  EXPECT_EQ(tokens[2], -1);

  const std::vector<float> values = {-10.0f, -0.25f, 0.0f, 0.25f, 0.75f, 10.0f};
  std::vector<int8_t> quantized(values.size());
  quantize_symmetric_int8_f32(values.size(), 0.05f, values.data(),
                              quantized.data());
  EXPECT_EQ(quantized[0], -128);
  EXPECT_EQ(quantized[1], -5);
  EXPECT_EQ(quantized[2], 0);
  EXPECT_EQ(quantized[3], 5);
  EXPECT_EQ(quantized[4], 15);
  EXPECT_EQ(quantized[5], 127);
}

TEST(LlmCompositesTest, TransposePackedSubbyteTile) {
    // 4 rows x 6 logical columns of 4-bit values (3 bytes/row).
    const std::vector<uint8_t> src = {
        0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB, 0xED, 0x1F, 0x32, 0x54, 0x76, 0x98,
    };
    std::vector<uint8_t> dst(4 * 2, 0); // 4 cols (col 1..4) x (4 rows / 2 = 2 bytes)
    transpose_packed_subbyte_tile(4, src.data(), 3, /*row_start=*/0, /*row_count=*/4, /*col_start=*/1,
                                  /*col_count=*/4, dst.data());
    // col 1 across rows 0..3: nibbles (2, 8, E, 5) -> bytes [0x82, 0x5E]
    EXPECT_EQ(dst[0], 0x82);
    EXPECT_EQ(dst[1], 0x5E);
    // col 2 across rows 0..3: nibbles (3, 9, F, 6) -> bytes [0x93, 0x6F]
    EXPECT_EQ(dst[2], 0x93);
    EXPECT_EQ(dst[3], 0x6F);

    // Partial-byte row_count = 3 (needs ceil(3 / 2) = 2 bytes per destination column, with sentinel guard bytes).
    std::vector<uint8_t> partial_dst(3 * 2 + 2, 0xAA);
    transpose_packed_subbyte_tile(4, src.data(), 3, /*row_start=*/0, /*row_count=*/3, /*col_start=*/1,
                                  /*col_count=*/3, partial_dst.data());
    // col 1 across rows 0..2: nibbles (2, 8, E) -> [0x82, 0x0E]
    EXPECT_EQ(partial_dst[0], 0x82);
    EXPECT_EQ(partial_dst[1], 0x0E);
    // col 2 across rows 0..2: nibbles (3, 9, F) -> [0x93, 0x0F]
    EXPECT_EQ(partial_dst[2], 0x93);
    EXPECT_EQ(partial_dst[3], 0x0F);
    // col 3 across rows 0..2: nibbles (4, A, 1) -> [0xA4, 0x01]
    EXPECT_EQ(partial_dst[4], 0xA4);
    EXPECT_EQ(partial_dst[5], 0x01);
    // Sentinel guard bytes past col_count * ceil(row_count / 2) must remain untouched.
    EXPECT_EQ(partial_dst[6], 0xAA);
    EXPECT_EQ(partial_dst[7], 0xAA);
}

TEST(LlmCompositesTest, GatedActivations) {
  const size_t rows = 3, width = 19;
  std::vector<float> input(rows * 2 * width);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.35f;
  }
  std::vector<float> silu_out(rows * width, 0.0f);
  std::vector<float> gelu_out(rows * width, 0.0f);
  ASSERT_EQ(silu_gate_f32(rows, width, input.data(), silu_out.data(), 0, rows),
            ynn_status_success);
  ASSERT_EQ(gelu_gate_f32(rows, width, input.data(), gelu_out.data(), 0, rows),
            ynn_status_success);

  constexpr float kSqrt2OverPi = 0.79788456080286535588f;
  for (size_t r = 0; r < rows; ++r) {
    const float* gate = input.data() + r * 2 * width;
    const float* up = gate + width;
    for (size_t c = 0; c < width; ++c) {
      const float g = gate[c];
      const float expected_silu = (g / (1.0f + std::exp(-g))) * up[c];
      const float expected_gelu =
          0.5f * g *
          (1.0f + std::tanh(kSqrt2OverPi * (g + 0.044715f * g * g * g))) *
          up[c];
      EXPECT_NEAR(silu_out[r * width + c], expected_silu, 1e-4f);
      EXPECT_NEAR(gelu_out[r * width + c], expected_gelu, 1e-4f);
    }
  }
}

TEST(LlmCompositesTest, RmsNormRotaryAndSlicedRmsRotary) {
  const size_t rows = 2, heads = 2, width = 10, half_w = width / 2;
  std::vector<float> x(rows * heads * width);
  std::vector<float> residual(x.size());
  std::vector<float> weight(width);
  std::vector<float> cos_v(rows * half_w), sin_v(rows * half_w);
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = 0.1f * static_cast<float>((i % 11) + 1);
    residual[i] = -0.05f * static_cast<float>(i % 7);
  }
  for (size_t i = 0; i < width; ++i)
    weight[i] = 0.5f + 0.1f * static_cast<float>(i);
  for (size_t i = 0; i < cos_v.size(); ++i) {
    const float angle = 0.2f * static_cast<float>(i);
    cos_v[i] = std::cos(angle);
    sin_v[i] = std::sin(angle);
  }

  std::vector<float> norm_out(x.size(), 0.0f);
  std::vector<float> rot_out(x.size(), 0.0f);
  std::vector<float> fused_out(x.size(), 0.0f);
  rms_norm_f32(rows * heads, width, 1e-5f, x.data(), weight.data(), nullptr,
               false, 1.0f, norm_out.data(), 0, rows * heads);
  rotary_f32(rows, heads, width, true, norm_out.data(), cos_v.data(),
             sin_v.data(), rot_out.data(), 0, rows);
  rms_rotary_f32(rows, heads, width, /*sliced=*/false, rows, heads * width, 0,
                 0, 1e-5f,
                 /*shared_angles=*/true, x.data(), weight.data(), cos_v.data(),
                 sin_v.data(), fused_out.data(), 0, rows * heads);
  for (size_t i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(fused_out[i], rot_out[i], 1e-6f);
  }

  // Residual + scale RMS norm check.
  std::vector<float> res_out(x.size(), 0.0f);
  rms_norm_f32(rows * heads, width, 1e-5f, x.data(), weight.data(),
               residual.data(), true, 0.5f, res_out.data(), 0, rows * heads);
  for (size_t i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(res_out[i], (residual[i] + norm_out[i]) * 0.5f, 1e-6f);
  }
}

TEST(LlmCompositesTest, DequantizeEmbeddingVariants) {
  const size_t vocab = 4, width = 8, groups = 2;
  // 4-bit asymmetric signed nibbles (2 nibbles/byte -> 4 bytes/row).
  std::vector<uint8_t> weights(vocab * (width / 2));
  std::vector<half> scales(vocab * groups);
  std::vector<half> zeros(vocab * groups);
  for (size_t i = 0; i < weights.size(); ++i)
    weights[i] = static_cast<uint8_t>((i * 37 + 13) & 0xFF);
  for (size_t i = 0; i < scales.size(); ++i) {
      scales[i] = half(0.125f * static_cast<float>((i % 3) + 1));
      zeros[i] = half(-0.25f * static_cast<float>(i % 2));
  }
  // Include positive and negative FP16 subnormals to verify non-ARM scalar conversion preserves 2^-24 mantissas.
  scales[0] = half::from_bits(0x0001u); // +2^-24
  zeros[0] = half::from_bits(0x8002u);  // -2^-23
  const std::vector<int32_t> tokens = {3, 0, 2};
  std::vector<float> out(tokens.size() * width, 0.0f);
  dequantize_embedding_f32(tokens.data(), tokens.size(), weights.data(), /*weights_bf16=*/false,
                           /*packed_bits=*/4, /*unsigned_packed=*/false, width, groups, scales.data(),
                           /*scales_fp16=*/true, zeros.data(), /*zeros_fp16=*/true,
                           /*multiplier=*/2.0f, out.data(), 0, tokens.size());
  for (size_t r = 0; r < tokens.size(); ++r) {
      const size_t tok = static_cast<size_t>(tokens[r]);
      for (size_t c = 0; c < width; ++c) {
          const size_t off = tok * width + c;
          const uint8_t raw = (weights[off / 2] >> ((off % 2) * 4)) & 0xF;
          const int val = (static_cast<int>(raw) ^ 8) - 8;
          const size_t g = tok * groups + c / (width / groups);
          const float expected =
              (static_cast<float>(val) * static_cast<float>(scales[g]) + static_cast<float>(zeros[g])) * 2.0f;
          EXPECT_NEAR(out[r * width + c], expected, 1e-5f);
      }
  }
  // Explicitly verify row 1 (token 0, group 0) retained non-zero subnormal contribution rather than flushing to 0.
  EXPECT_GT(std::abs(out[1 * width + 0]), 0.0f);
}

TEST(LlmCompositesTest, AttentionF32Gqa8AndGeneral) {
    const size_t total_keys = 6, head_dim = 12;
    std::vector<float> query(8 * head_dim);
    std::vector<float> key(total_keys * head_dim);
    std::vector<float> value(total_keys * head_dim);
    std::vector<float> mask = {0.0f, -0.2f, 0.1f, -1e9f, -1e9f, -1e9f};
    for (size_t i = 0; i < query.size(); ++i) query[i] = 0.05f * static_cast<float>(static_cast<int>(i % 9) - 4);
    for (size_t i = 0; i < key.size(); ++i) {
        key[i] = 0.07f * static_cast<float>(static_cast<int>(i % 11) - 5);
        value[i] = 0.1f * static_cast<float>(static_cast<int>(i % 7) - 3);
    }

    attention_f32_params fast_params{};
    fast_params.batch_size = 1;
    fast_params.query_length = 1;
    fast_params.query_heads = 8;
    fast_params.key_value_heads = 1;
    fast_params.key_start = 0;
    fast_params.key_length = total_keys;
    fast_params.total_keys = total_keys;
    fast_params.head_dim = head_dim;
    fast_params.mask_key_stride = 1;
    fast_params.scale = 0.25f;

    const size_t fast_items = attention_f32_work_items(fast_params);
    ASSERT_EQ(fast_items, 4u);
    std::vector<float> fast_scratch(attention_f32_scratch_size(fast_params));
    std::vector<float> fast_out(8 * head_dim, -1.0f);
    attention_f32(fast_params, query.data(), key.data(), value.data(), mask.data(), fast_out.data(), 0, fast_items,
                  fast_scratch.data());

    // Also verify single-item invocation across each of the 4 head-pair items matches full-range execution.
    std::vector<float> split_out(8 * head_dim, -3.0f);
    for (size_t item = 0; item < fast_items; ++item) {
        std::vector<float> item_scratch(attention_f32_scratch_size(fast_params), -99.0f);
        attention_f32(fast_params, query.data(), key.data(), value.data(), mask.data(), split_out.data(), item,
                      item + 1, item_scratch.data());
    }
    for (size_t i = 0; i < fast_out.size(); ++i) {
        EXPECT_FLOAT_EQ(split_out[i], fast_out[i]);
    }

    // Force general path by setting mask_head_stride != 0 with identical broadcast mask rows.
    std::vector<float> head_mask(8 * total_keys);
    for (size_t h = 0; h < 8; ++h) {
        std::copy(mask.begin(), mask.end(), head_mask.begin() + h * total_keys);
    }
    attention_f32_params gen_params = fast_params;
    gen_params.mask_head_stride = total_keys;
    ASSERT_EQ(attention_f32_work_items(gen_params), 8u);
    std::vector<float> gen_scratch(attention_f32_scratch_size(gen_params));
    std::vector<float> gen_out(8 * head_dim, -2.0f);
    attention_f32(gen_params, query.data(), key.data(), value.data(), head_mask.data(), gen_out.data(), 0, 8,
                  gen_scratch.data());

    for (size_t i = 0; i < fast_out.size(); ++i) {
        EXPECT_NEAR(fast_out[i], gen_out[i], 1e-5f);
    }
}

TEST(LlmCompositesTest, BlockwiseLinearInt4AndInt8) {
  for (int32_t bits : {4, 8}) {
    if (!is_blockwise_linear_supported(bits)) continue;
    for (size_t block_size : {size_t{32}, size_t{128}}) {
      for (bool has_zero : {false, true}) {
        if (bits == 8 && has_zero) continue;
        const size_t N = 67, K = 256, B = K / block_size;
        std::vector<int8_t> w_unpacked(N * K);
        std::vector<uint8_t> w_packed(bits == 4 ? (N * K / 2) : (N * K));
        for (size_t i = 0; i < N * K; ++i) {
          if (bits == 4) {
            const int v = static_cast<int>(i % 15) - 7;
            w_unpacked[i] = static_cast<int8_t>(v);
            const uint8_t nib = static_cast<uint8_t>(v & 0xF);
            if ((i & 1) == 0)
              w_packed[i / 2] = nib;
            else
              w_packed[i / 2] |= static_cast<uint8_t>(nib << 4);
          } else {
            const int v = static_cast<int>(i % 127) - 63;
            w_unpacked[i] = static_cast<int8_t>(v);
            w_packed[i] = static_cast<uint8_t>(static_cast<int8_t>(v));
          }
        }
        std::vector<float> scales(N * B);
        std::vector<float> zeros(N * B, 0.0f);
        for (size_t i = 0; i < N * B; ++i) {
          scales[i] = 0.02f * static_cast<float>((i % 5) + 1);
          if (has_zero)
            zeros[i] = -0.03f * static_cast<float>(static_cast<int>(i % 3) - 1);
        }

        const size_t tiles = blockwise_linear_tile_count(N);
        std::vector<uint8_t> tiles_buf(
            packed_blockwise_linear_size(bits, N, K, block_size, has_zero));
        pack_blockwise_linear_tiles(
            bits, N, K, block_size, w_packed.data(), scales.data(),
            /*scales_fp16=*/false, has_zero ? zeros.data() : nullptr,
            /*zeros_fp16=*/false, 0, tiles, tiles_buf.data());

        for (size_t M :
             {size_t{1}, size_t{2}, size_t{3}, size_t{4}, size_t{35}}) {
          std::vector<float> x(M * K);
          for (size_t i = 0; i < x.size(); ++i) {
            // Skewed positive range [-0.2, 3.4] exercises qd8 asymmetric
            // activation zero-point correction for bits == 8.
            x[i] = 0.2f * static_cast<float>(static_cast<int>(i % 19) - 1);
          }
          const bool use_bsum = has_zero || bits == 8;
          std::vector<int8_t> act_q(M * K);
          std::vector<float> act_s(M * B);
          std::vector<float> act_b(use_bsum ? M * B : 0);
          quantize_blockwise_activations_f32(
              bits, M, K, block_size, use_bsum, x.data(), act_q.data(),
              act_s.data(), use_bsum ? act_b.data() : nullptr, 0, M);
          std::vector<float> y(M * N, 0.0f);
          blockwise_linear_f32(
              bits, M, N, K, block_size, has_zero, /*output_scale=*/0.0f,
              tiles_buf.data(), act_q.data(), act_s.data(),
              use_bsum ? act_b.data() : nullptr, y.data(), 0, tiles);

          // Compare against float reference using quantized activations +
          // FP16/FP32 scales.
          for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
              float ref = 0.0f;
              for (size_t b = 0; b < B; ++b) {
                float rmin = 0.0f, rmax = 0.0f, amax = 0.0f, bsum = 0.0f;
                for (size_t k = 0; k < block_size; ++k) {
                  const float xv = x[(m * B + b) * block_size + k];
                  rmin = std::min(rmin, xv);
                  rmax = std::max(rmax, xv);
                  amax = std::max(amax, std::abs(xv));
                  bsum += xv;
                }
                float a_scale = 0.0f;
                int32_t dot = 0;
                if (bits == 8) {
                  const float rdiff = rmax - rmin;
                  const float inv = rdiff > 0.0f ? 255.0f / rdiff : 0.0f;
                  a_scale = rdiff > 0.0f ? rdiff / 255.0f : 1.0f;
                  int32_t zp = -128;
                  if (rdiff > 0.0f) {
                    const float min_error = -128.0f + rmin * inv;
                    const float max_error = 127.0f + rmax * inv;
                    const float fzp = min_error + max_error > 0.0f
                                          ? -128.0f - rmin * inv
                                          : 127.0f - rmax * inv;
                    zp = static_cast<int32_t>(
                        std::rint(std::clamp(fzp, -128.0f, 127.0f)));
                  }
                  for (size_t k = 0; k < block_size; ++k) {
                    const int32_t qv = std::clamp<int32_t>(
                        static_cast<int32_t>(
                            std::round(x[(m * B + b) * block_size + k] * inv)) +
                            zp,
                        -128, 127);
                    dot +=
                        (qv - zp) * static_cast<int32_t>(
                                        w_unpacked[n * K + b * block_size + k]);
                  }
                } else {
                  a_scale = amax / 127.0f;
                  const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
                  for (size_t k = 0; k < block_size; ++k) {
                    const int32_t qv = static_cast<int32_t>(
                        std::nearbyint(x[(m * B + b) * block_size + k] * inv));
                    dot += qv * static_cast<int32_t>(
                                    w_unpacked[n * K + b * block_size + k]);
                  }
                }
                const float ws =
                    bits == 4 ? static_cast<float>(half(scales[n * B + b]))
                              : scales[n * B + b];
                ref += static_cast<float>(dot) * (ws * a_scale);
                if (has_zero) {
                  const float wz = static_cast<float>(half(zeros[n * B + b]));
                  ref += bsum * wz;
                }
              }
              EXPECT_NEAR(y[m * N + n], ref, 1e-3f)
                  << "bits=" << bits << " block=" << block_size
                  << " zero=" << has_zero << " M=" << M << " m=" << m
                  << " n=" << n;
            }
          }
        }
      }
    }
  }
}

TEST(LlmCompositesTest, SymmetricLinearAndGatedMultiply) {
    const size_t elems = 1037;
    std::vector<float> gate(elems), up(elems), silu_out(elems, 0.0f), gelu_out(elems, 0.0f);
    for (size_t i = 0; i < elems; ++i) {
        gate[i] = 0.25f * static_cast<float>(static_cast<int>(i % 15) - 7);
        up[i] = 0.15f * static_cast<float>(static_cast<int>(i % 11) - 5);
    }
    const size_t g_tiles = gated_multiply_tile_count(elems);
    ASSERT_EQ(silu_multiply_tiles_f32(elems, gate.data(), up.data(), silu_out.data(), 0, g_tiles), ynn_status_success);
    ASSERT_EQ(gelu_multiply_tiles_f32(elems, gate.data(), up.data(), gelu_out.data(), 0, g_tiles), ynn_status_success);
    constexpr float kSqrt2OverPi = 0.79788456080286535588f;
    for (size_t i = 0; i < elems; ++i) {
        const float g = gate[i];
        const float exp_silu = (g / (1.0f + std::exp(-g))) * up[i];
        const float exp_gelu = 0.5f * g * (1.0f + std::tanh(kSqrt2OverPi * (g + 0.044715f * g * g * g))) * up[i];
        EXPECT_NEAR(silu_out[i], exp_silu, 1e-4f);
        EXPECT_NEAR(gelu_out[i], exp_gelu, 1e-4f);
    }

    for (int32_t bits : {2, 4, 8}) {
        if (!is_symmetric_linear_supported(bits)) continue;
        const size_t vals_per_byte = static_cast<size_t>(8 / bits);
        const size_t n_cols = 37, k_cols = 72; // non-multiples of 32
        const float in_scale = 0.125f, out_scale = 0.0625f;
        std::vector<int8_t> w_unpacked(n_cols * k_cols);
        std::vector<uint8_t> w_packed(n_cols * (k_cols / vals_per_byte), 0);
        const int low = -(1 << (bits - 1));
        const int span = 1 << bits;
        const uint8_t mask = static_cast<uint8_t>(span - 1);
        for (size_t r = 0; r < n_cols; ++r) {
            for (size_t c = 0; c < k_cols; ++c) {
                const int v = low + static_cast<int>((r * 13 + c * 7) % (span - 1)) + 1;
                w_unpacked[r * k_cols + c] = static_cast<int8_t>(v);
                const uint8_t code = static_cast<uint8_t>(v) & mask;
                w_packed[r * (k_cols / vals_per_byte) + c / vals_per_byte] |=
                    static_cast<uint8_t>(code << ((c % vals_per_byte) * bits));
            }
        }
        std::vector<float> w_scales(n_cols);
        for (size_t r = 0; r < n_cols; ++r) w_scales[r] = 0.0078125f * static_cast<float>((r % 5) + 1);

        const size_t tiles = blockwise_linear_tile_count(n_cols);
        std::vector<uint8_t> packed_buf(packed_symmetric_linear_size(bits, n_cols, k_cols, k_cols));
        pack_symmetric_linear_tiles(bits, n_cols, k_cols, 0, k_cols, w_packed.data(), w_scales.data(), false, 0, tiles,
                                    packed_buf.data());

        for (size_t m_rows : {size_t{1}, size_t{5}}) {
            std::vector<float> x(m_rows * k_cols);
            for (size_t i = 0; i < x.size(); ++i) x[i] = 0.125f * static_cast<float>(static_cast<int>(i % 17) - 8);
            const size_t padded_k = symmetric_linear_padded_block(k_cols);
            std::vector<int8_t> act_q(m_rows * padded_k);
            std::vector<float> act_s(m_rows);
            quantize_symmetric_blocks_f32(bits, m_rows, k_cols, k_cols, in_scale, x.data(), act_q.data(), act_s.data(),
                                          0, m_rows);
            std::vector<float> y(m_rows * n_cols, 0.0f);
            symmetric_linear_f32(bits, m_rows, n_cols, k_cols, k_cols, out_scale, packed_buf.data(), act_q.data(),
                                 act_s.data(), y.data(), n_cols, 0, tiles);

            for (size_t m = 0; m < m_rows; ++m) {
                for (size_t n = 0; n < n_cols; ++n) {
                    int32_t dot = 0;
                    for (size_t k = 0; k < k_cols; ++k) {
                        const int32_t qx = static_cast<int32_t>(
                            std::clamp(std::nearbyint(x[m * k_cols + k] / in_scale), -128.0f, 127.0f));
                        dot += qx * static_cast<int32_t>(w_unpacked[n * k_cols + k]);
                    }
                    const float raw_y = static_cast<float>(dot) * (w_scales[n] * in_scale);
                    const float ref = std::clamp(std::nearbyint(raw_y / out_scale), -128.0f, 127.0f) * out_scale;
                    EXPECT_NEAR(y[m * n_cols + n], ref, 1e-5f) << "bits=" << bits << " m=" << m << " n=" << n;
                }
            }
        }
    }
}

}  // namespace
}  // namespace ynn
