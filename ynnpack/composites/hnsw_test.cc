// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/composites/hnsw.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "slinky/base/thread_pool_impl.h"

namespace ynn {
namespace {

uint16_t f32_to_bf16_trunc(float v) {
  uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

float bf16_to_f32(uint16_t b) {
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

TEST(HnswCompositesTest, DenseDotMicrokernelsMatchScalar) {
  for (const bool bf16 : {false, true}) {
    for (const size_t width : {size_t{4}, size_t{64}, size_t{67}, size_t{153}}) {
      std::mt19937 rng(123);
      std::vector<float> q_f32(width);
      std::vector<float> rows_f32(5 * width);
      std::vector<uint16_t> rows_bf16(5 * width);
      for (size_t i = 0; i < width; ++i) {
        q_f32[i] = (static_cast<float>(rng() % 101) - 50.0f) / 16.0f;
      }
      for (size_t i = 0; i < rows_f32.size(); ++i) {
        const float v = (static_cast<float>(rng() % 65) - 32.0f) / 8.0f;
        rows_bf16[i] = f32_to_bf16_trunc(v);
        rows_f32[i] = bf16 ? bf16_to_f32(rows_bf16[i]) : v;
      }
      const hnsw_dot_kernels k = get_hnsw_dot_kernels(bf16);
      const auto row_ptr = [&](size_t r) -> const void* {
        return bf16 ? static_cast<const void*>(rows_bf16.data() + r * width)
                    : static_cast<const void*>(rows_f32.data() + r * width);
      };

      float d4[4] = {};
      k.dot4(q_f32.data(), row_ptr(0), row_ptr(1), row_ptr(2), row_ptr(3), width,
             d4);
      float rd4[4] = {};
      k.row_dot4(row_ptr(4), row_ptr(0), row_ptr(1), row_ptr(2), row_ptr(3),
                 width, rd4);
      for (size_t r = 0; r < 4; ++r) {
        double ref_q = 0.0, ref_row = 0.0;
        for (size_t c = 0; c < width; ++c) {
          ref_q += static_cast<double>(q_f32[c]) * rows_f32[r * width + c];
          ref_row +=
              static_cast<double>(rows_f32[4 * width + c]) *
              rows_f32[r * width + c];
        }
        EXPECT_NEAR(k.dot(q_f32.data(), row_ptr(r), width),
                    static_cast<float>(ref_q), 1e-3f);
        EXPECT_NEAR(d4[r], static_cast<float>(ref_q), 1e-3f);
        EXPECT_NEAR(k.row_dot(row_ptr(4), row_ptr(r), width),
                    static_cast<float>(ref_row), 1e-3f);
        EXPECT_NEAR(rd4[r], static_cast<float>(ref_row), 1e-3f);
      }
    }
  }
}

TEST(HnswCompositesTest, DeterministicBuildAndThreadPoolParity) {
  constexpr size_t rows = 256, width = 67, k = 8;
  std::mt19937 rng(17);
  std::vector<float> values(rows * width);
  std::vector<uint16_t> encoded(rows * width);
  for (size_t i = 0; i < values.size(); ++i) {
    const float v = (static_cast<float>(rng() % 33) - 16.0f) / 16.0f;
    encoded[i] = f32_to_bf16_trunc(v);
    values[i] = bf16_to_f32(encoded[i]);
  }

  hnsw_build_options opts;
  opts.m = 12;
  opts.ef_construction = 128;
  opts.seed = 42;

  size_t last_progress = 0;
  hnsw_graph g_single;
  ASSERT_EQ(build_hnsw_graph(
                encoded.data(), /*vectors_bf16=*/true, rows, width, opts,
                &g_single,
                [&](size_t done, size_t total) {
                  EXPECT_LE(done, total);
                  EXPECT_GT(done, last_progress);
                  last_progress = done;
                }),
            ynn_status_success);
  EXPECT_EQ(last_progress, rows);
  EXPECT_TRUE(validate_hnsw_graph(g_single));

  slinky::thread_pool_impl pool(4);
  hnsw_graph g_threaded;
  ASSERT_EQ(build_hnsw_graph(
                encoded.data(), /*vectors_bf16=*/true, rows, width, opts,
                &g_threaded, reinterpret_cast<ynn_threadpool_t>(&pool)),
            ynn_status_success);
  EXPECT_EQ(g_single.entry, g_threaded.entry);
  EXPECT_EQ(g_single.labels, g_threaded.labels);
  EXPECT_EQ(g_single.levels, g_threaded.levels);
  EXPECT_EQ(g_single.base_offsets, g_threaded.base_offsets);
  EXPECT_EQ(g_single.base_edges, g_threaded.base_edges);
  ASSERT_EQ(g_single.upper.size(), g_threaded.upper.size());
  for (size_t l = 0; l < g_single.upper.size(); ++l) {
    EXPECT_EQ(g_single.upper[l].nodes, g_threaded.upper[l].nodes);
    EXPECT_EQ(g_single.upper[l].offsets, g_threaded.upper[l].offsets);
    EXPECT_EQ(g_single.upper[l].edges, g_threaded.upper[l].edges);
  }

  hnsw_search_index search_idx;
  ASSERT_EQ(prepare_hnsw_dense_search(g_single, encoded.data(),
                                      /*vectors_bf16=*/true, &search_idx),
            ynn_status_success);
  hnsw_workspace ws;
  std::vector<float> query(width);
  for (size_t c = 0; c < width; ++c) {
    query[c] = (static_cast<float>(rng() % 101) - 50.0f) / 32.0f;
  }
  const hnsw_neighbor* found = nullptr;
  size_t found_count = 0;
  ASSERT_EQ(search_hnsw(search_idx, query.data(), width, k, /*ef=*/rows, &ws,
                        &found, &found_count,
                        reinterpret_cast<ynn_threadpool_t>(&pool)),
            ynn_status_success);
  ASSERT_EQ(found_count, k);

  std::vector<std::pair<float, uint32_t>> exact;
  for (size_t r = 0; r < rows; ++r) {
    double dot = 0.0;
    for (size_t c = 0; c < width; ++c) {
      dot += static_cast<double>(query[c]) * values[r * width + c];
    }
    exact.emplace_back(1.0f - static_cast<float>(dot), static_cast<uint32_t>(r));
  }
  std::sort(exact.begin(), exact.end());
  EXPECT_EQ(found[0].id, exact[0].second);
  EXPECT_NEAR(found[0].distance, exact[0].first, 1e-4f);
}

TEST(HnswCompositesTest, QuantizedSearchMatchesSlices4AndScalar) {
  constexpr size_t rows = 16, width = 128, group = 32, blocks = width / group;
  std::mt19937 rng(29);
  std::vector<uint8_t> packed(rows * width / 2, 0);
  std::vector<float> scales(rows * blocks, 0.25f);
  std::vector<float> zeros(rows * blocks, -0.125f);
  std::vector<float> decoded(rows * width, 0.0f);
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < width; ++c) {
      const uint32_t enc = rng() % 16u;
      const int32_t val = static_cast<int32_t>(enc) - (enc >= 8u ? 16 : 0);
      packed[r * (width / 2) + c / 2] |=
          static_cast<uint8_t>(enc << (4 * (c % 2)));
      decoded[r * width + c] =
          val * scales[r * blocks + c / group] + zeros[r * blocks + c / group];
    }
  }
  hnsw_build_options opts{8, 32, 42};
  hnsw_graph graph;
  ASSERT_EQ(build_hnsw_graph(decoded.data(), /*vectors_bf16=*/false, rows, width,
                             opts, &graph),
            ynn_status_success);
  hnsw_search_index qidx;
  ASSERT_EQ(prepare_hnsw_quantized_search(
                graph, packed.data(), scales.data(), /*scale_fp16=*/false,
                zeros.data(), /*zero_fp16=*/false, /*bits=*/4,
                static_cast<int32_t>(group), /*input_scale=*/0.0f, &qidx),
            ynn_status_success);

  std::vector<float> query(width);
  for (float& v : query) v = (static_cast<float>(rng() % 81) - 40.0f) / 16.0f;
  hnsw_workspace ws;
  const hnsw_neighbor* found = nullptr;
  size_t count = 0;
  ASSERT_EQ(search_hnsw(qidx, query.data(), width, rows, rows, &ws, &found,
                        &count),
            ynn_status_success);
  ASSERT_EQ(count, rows);

  float s4[4] = {};
  const uint8_t* base = qidx.packed_rows.data();
  const size_t sb = qidx.slice_bytes;
  hnsw_quantized_slices4_dot(base + 0 * sb, base + 1 * sb, base + 2 * sb,
                             base + 3 * sb, width, 4,
                             static_cast<int32_t>(group), false, false, true,
                             ws.q_i8.data(), ws.q_scale.data(),
                             ws.q_bsum.data(), s4);
  for (size_t i = 0; i < 4; ++i) {
    const float s1 = hnsw_quantized_slice_dot(
        base + i * sb, width, 4, static_cast<int32_t>(group), false, false,
        true, ws.q_i8.data(), ws.q_scale.data(), ws.q_bsum.data());
    EXPECT_NEAR(s4[i], s1, 1e-5f);
  }
}

}  // namespace
}  // namespace ynn
