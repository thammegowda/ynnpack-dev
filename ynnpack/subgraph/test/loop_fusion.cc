// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Tests that the automatic scheduler successfully fuses loops between operators
// in small subgraphs. We verify this by using a custom allocator to track
// max_allocation_size during execution; when loops are fused, intermediate
// buffers (like packed weights or elementwise outputs) are processed and
// allocated per-block inside the loop nest rather than as full buffers up
// front.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/composites/composites.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {
namespace {

class LoopFusionTest : public testing::Test {
 protected:
  void MakeRuntime(ynn_subgraph_t subgraph) {
    ynn_threadpool_t threadpool = nullptr;
    ASSERT_EQ(ynn_create_threadpool(TestScheduler::scheduler(), &scheduler_,
                                    /*flags=*/0, &threadpool),
              ynn_status_success);
    threadpool_.reset(threadpool);
    ASSERT_EQ(ynn_optimize_subgraph(subgraph, threadpool, /*flags=*/0),
              ynn_status_success);
    ynn_runtime_t runtime = nullptr;
    ASSERT_EQ(ynn_create_runtime(subgraph, threadpool, /*flags=*/0, &runtime),
              ynn_status_success);
    runtime_.reset(runtime);

    // Disable block reuse so every allocation reaches this hook.
    runtime_->eval_config.use_memory_pool = false;
    runtime_->eval_config.allocate = [this](std::size_t size,
                                            std::size_t alignment) {
      void* ptr = slinky::allocate_bytes(size, alignment);
      if (ptr) {
        max_allocation_size_ = std::max(max_allocation_size_, size);
      }
      return ptr;
    };
  }

  void ReshapeExternalTensor(uint32_t id, const std::vector<size_t>& shape,
                             void* data) {
    ASSERT_EQ(ynn_set_external_value_shape(runtime_.get(), id, shape.size(),
                                           shape.data()),
              ynn_status_success);
    ASSERT_EQ(ynn_set_external_value_data(runtime_.get(), id, data),
              ynn_status_success);
  }

  void SetupExternalTensor(uint32_t id, void* data) {
    ASSERT_EQ(ynn_set_external_value_data(runtime_.get(), id, data),
              ynn_status_success);
  }

  void RunPipeline() {
    max_allocation_size_ = 0;
    ASSERT_EQ(ynn_reshape_runtime(runtime_.get()), ynn_status_success);
    ASSERT_EQ(ynn_invoke_runtime(runtime_.get()), ynn_status_success);
  }

  TestScheduler scheduler_{3};
  std::unique_ptr<ynn_threadpool, decltype(&ynn_delete_threadpool)> threadpool_{
      nullptr, ynn_delete_threadpool};
  std::unique_ptr<ynn_runtime, decltype(&ynn_delete_runtime)> runtime_{
      nullptr, ynn_delete_runtime};
  std::size_t max_allocation_size_ = 0;
};

// pack_b should be computed inside the dot's loop nest, so packing happens
// per-block instead of materializing the whole packed buffer up front.
TEST_F(LoopFusionTest, PackFusesWithDot) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t out_id = 2;
  SubgraphBuilder subgraph(3);
  subgraph.AddInput(type_of<float>(), TensorShape(2), a_id)
      .AddInput(type_of<float>(), TensorShape(2), b_id)
      .AddOutput(type_of<float>(), TensorShape(2), out_id)
      .AddDot(1, a_id, b_id, YNN_INVALID_VALUE_ID, out_id);

  MakeRuntime(subgraph.GetSubgraph());

  Tensor<float> a({16, 512});
  Tensor<float> b({512, 1024});
  Tensor<float> out({16, 1024});
  ReshapeExternalTensor(a_id, {16, 512}, a.data());
  ReshapeExternalTensor(b_id, {512, 1024}, b.data());
  SetupExternalTensor(out_id, out.data());
  RunPipeline();
  EXPECT_LT(max_allocation_size_, b.size_bytes());
}

// The pipeline is dot(A, pack(exp(B))). exp's natural loop order does not
// match the dot's loop nest positionally (its n dimension is innermost, while
// the dot's nest iterates n outermost), so fusing it requires the scheduler
// to match loop splits by source region rather than by position.
TEST_F(LoopFusionTest, ProducerOfPackedInputFusesWithDot) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t out_id = 2;
  uint32_t exp_id = YNN_INVALID_VALUE_ID;
  SubgraphBuilder subgraph(3);
  subgraph.AddInput(type_of<float>(), TensorShape(2), a_id)
      .AddInput(type_of<float>(), TensorShape(2), b_id)
      .AddOutput(type_of<float>(), TensorShape(2), out_id)
      .AddTensor(type_of<float>(), TensorShape(2), exp_id)
      .AddUnary(ynn_unary_exp, b_id, exp_id)
      .AddDot(1, a_id, exp_id, YNN_INVALID_VALUE_ID, out_id);

  MakeRuntime(subgraph.GetSubgraph());

  Tensor<float> a({16, 512});
  Tensor<float> b({512, 1024});
  Tensor<float> out({16, 1024});
  ReshapeExternalTensor(a_id, {16, 512}, a.data());
  ReshapeExternalTensor(b_id, {512, 1024}, b.data());
  SetupExternalTensor(out_id, out.data());
  RunPipeline();
  EXPECT_LT(max_allocation_size_, b.size_bytes());
}

// The pipeline is dot(A, transpose(exp(Bt))). The transpose is folded into the
// packing (always_alias_transpose), so the func chain is exp -> transpose
// (aliased copy) -> pack_b -> dot. In this layout exp's loop order matches the
// dot's loop nest positionally, so fusion of exp is blocked *only* by the
// source region inference breaking at pack's non-identity input bounds.
TEST_F(LoopFusionTest, ProducerOfTransposedPackedInputFusesWithDot) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t out_id = 2;
  uint32_t exp_id = YNN_INVALID_VALUE_ID;
  uint32_t transpose_id = YNN_INVALID_VALUE_ID;
  SubgraphBuilder subgraph(3);
  subgraph.AddInput(type_of<float>(), TensorShape(2), a_id)
      .AddInput(type_of<float>(), TensorShape(2), b_id)
      .AddOutput(type_of<float>(), TensorShape(2), out_id)
      .AddTensor(type_of<float>(), TensorShape(2), exp_id)
      .AddTensor(type_of<float>(), TensorShape(2), transpose_id)
      .AddUnary(ynn_unary_exp, b_id, exp_id)
      .AddTranspose({1, 0}, exp_id, transpose_id)
      .AddDot(1, a_id, transpose_id, YNN_INVALID_VALUE_ID, out_id);

  MakeRuntime(subgraph.GetSubgraph());

  Tensor<float> a({16, 512});
  Tensor<float> b({1024, 512});
  Tensor<float> out({16, 1024});
  ReshapeExternalTensor(a_id, {16, 512}, a.data());
  ReshapeExternalTensor(b_id, {1024, 512}, b.data());
  SetupExternalTensor(out_id, out.data());
  RunPipeline();
  EXPECT_LT(max_allocation_size_, b.size_bytes());
}

// Two dots accumulated into one output: dot(A, B1, c=dot(A, B0)), like the
// dots of the dot_sum composite. The second dot fuses into the loops of the
// first one; both require specific tile steps for the shared loops, and the
// scheduler reconciles them with a least common multiple (lcm_sat).
TEST_F(LoopFusionTest, TwoDotsShareLoops) {
  const uint32_t a_id = 0;
  const uint32_t b0_id = 1;
  const uint32_t b1_id = 2;
  const uint32_t out_id = 3;
  uint32_t dot0_id = YNN_INVALID_VALUE_ID;
  SubgraphBuilder subgraph(4);
  subgraph.AddInput(type_of<float>(), TensorShape(2), a_id)
      .AddInput(type_of<float>(), TensorShape(2), b0_id)
      .AddInput(type_of<float>(), TensorShape(2), b1_id)
      .AddOutput(type_of<float>(), TensorShape(2), out_id)
      .AddTensor(type_of<float>(), TensorShape(2), dot0_id)
      .AddDot(1, a_id, b0_id, YNN_INVALID_VALUE_ID, dot0_id)
      .AddDot(1, a_id, b1_id, dot0_id, out_id);

  MakeRuntime(subgraph.GetSubgraph());

  const size_t M = 128, K = 256, N = 1024;
  Tensor<float> a({M, K});
  Tensor<float> b0({K, N});
  Tensor<float> b1({K, N});
  Tensor<float> out({M, N});
  a.fill(1.0f);
  b0.fill(1.0f);
  b1.fill(2.0f);
  ReshapeExternalTensor(a_id, {M, K}, a.data());
  ReshapeExternalTensor(b0_id, {K, N}, b0.data());
  ReshapeExternalTensor(b1_id, {K, N}, b1.data());
  SetupExternalTensor(out_id, out.data());
  RunPipeline();
  // Both dot intermediates should be computed per-block inside the shared
  // loop nest rather than materialized in full.
  EXPECT_LT(max_allocation_size_, M * N * sizeof(float));
  // The reconciled loop steps must still produce correct results.
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      ASSERT_EQ(out({i, j}), 3.0f * K) << i << " " << j;
    }
  }
}

TEST_F(LoopFusionTest, FeedForwardUsesBoundedScratch) {
  const size_t rows = 128, hidden = 64, intermediate = 4096, output_width = 32;
  const size_t tile_size = 2048;
  const size_t num_tiles = intermediate / tile_size;
  for (const bool gated : {false, true}) {
    SCOPED_TRACE(gated ? "gated" : "plain");
    Tensor<float> input({rows, hidden});
    Tensor<int8_t> gate_weight({num_tiles, hidden, tile_size});
    Tensor<float> gate_scale({num_tiles, tile_size, 1});
    Tensor<int8_t> up_weight({num_tiles, hidden, tile_size});
    Tensor<float> up_scale({num_tiles, tile_size, 1});
    Tensor<int8_t> down_weight({num_tiles, tile_size, output_width});
    Tensor<float> down_scale({output_width, 1});
    Tensor<float> output({rows, output_width});
    input.fill(1.0f);
    gate_weight.fill(1);
    gate_scale.fill(1.0f / hidden);
    up_weight.fill(1);
    up_scale.fill(1.0f / hidden);
    down_weight.fill(1);
    down_scale.fill(1.0f / intermediate);

    const uint32_t input_id = 0, output_id = 1;
    std::vector<uint32_t> gate_weight_ids(num_tiles, YNN_INVALID_VALUE_ID);
    std::vector<uint32_t> gate_scale_ids(num_tiles, YNN_INVALID_VALUE_ID);
    std::vector<uint32_t> up_weight_ids(num_tiles, YNN_INVALID_VALUE_ID);
    std::vector<uint32_t> up_scale_ids(num_tiles, YNN_INVALID_VALUE_ID);
    std::vector<uint32_t> down_weight_ids(num_tiles, YNN_INVALID_VALUE_ID);
    uint32_t down_scale_id = YNN_INVALID_VALUE_ID;
    uint32_t input_scale_id = YNN_INVALID_VALUE_ID;
    uint32_t first_output_scale_id = YNN_INVALID_VALUE_ID;
    uint32_t down_input_scale_id = YNN_INVALID_VALUE_ID;
    uint32_t down_output_scale_id = YNN_INVALID_VALUE_ID;
    const float input_scale = 1.0f;
    const float first_output_scale = 1.0f / 128.0f;
    const float down_input_scale = 1.0f / 128.0f;
    const float down_output_scale = 1.0f / 128.0f;
    SubgraphBuilder subgraph(2);
    subgraph.AddInput(ynn_type_fp32, TensorShape(2), input_id)
        .AddTensor(ynn_type_fp32, down_scale.extents(), down_scale_id,
                   down_scale.data())
        .AddTensor(ynn_type_fp32, TensorShape{}, input_scale_id, &input_scale,
                   YNN_VALUE_FLAG_COPY_DATA)
        .AddTensor(ynn_type_fp32, TensorShape{}, first_output_scale_id,
                   &first_output_scale, YNN_VALUE_FLAG_COPY_DATA)
        .AddTensor(ynn_type_fp32, TensorShape{}, down_input_scale_id,
                   &down_input_scale, YNN_VALUE_FLAG_COPY_DATA)
        .AddTensor(ynn_type_fp32, TensorShape{}, down_output_scale_id,
                   &down_output_scale, YNN_VALUE_FLAG_COPY_DATA)
        .AddOutput(ynn_type_fp32, TensorShape(2), output_id);
    const std::vector<size_t> first_shape{hidden, tile_size};
    const std::vector<size_t> first_scale_shape{tile_size, 1};
    const std::vector<size_t> down_shape{tile_size, output_width};
    for (size_t tile = 0; tile < num_tiles; ++tile) {
      subgraph
          .AddTensor(ynn_type_int8, first_shape, gate_weight_ids[tile],
                     gate_weight.data() + tile * hidden * tile_size)
          .AddTensor(ynn_type_fp32, first_scale_shape, gate_scale_ids[tile],
                     gate_scale.data() + tile * tile_size)
          .AddTensor(ynn_type_int8, down_shape, down_weight_ids[tile],
                     down_weight.data() + tile * tile_size * output_width);
      if (gated) {
        subgraph
            .AddTensor(ynn_type_int8, first_shape, up_weight_ids[tile],
                       up_weight.data() + tile * hidden * tile_size)
            .AddTensor(ynn_type_fp32, first_scale_shape, up_scale_ids[tile],
                       up_scale.data() + tile * tile_size);
      }
    }
    uint32_t actual_output_id = output_id;
    ASSERT_EQ(define_packed_feed_forward(
                  subgraph.GetSubgraph(), input_id, input_scale_id,
                  gate_weight_ids.data(), gate_scale_ids.data(),
                  gated ? up_weight_ids.data() : nullptr,
                  gated ? up_scale_ids.data() : nullptr, first_output_scale_id,
                  down_weight_ids.data(), down_scale_id, down_input_scale_id,
                  down_output_scale_id, hidden, intermediate, tile_size, gated,
                  actual_output_id),
              ynn_status_success);
    ASSERT_EQ(actual_output_id, output_id);

    MakeRuntime(subgraph.GetSubgraph());
    ReshapeExternalTensor(input_id, {rows, hidden}, input.data());
    SetupExternalTensor(output_id, output.data());
    RunPipeline();
    EXPECT_LE(max_allocation_size_, rows * tile_size * sizeof(float));
    const auto calibrate = [](float value, float scale) {
      return std::clamp(std::nearbyint(value / scale), -128.0f, 127.0f) * scale;
    };
    const float gate = calibrate(1.0f, 1.0f / 128.0f);
    const float gelu =
        gate * 0.5f *
        (1.0f + std::tanh(std::sqrt(2.0f / M_PI) *
                          (gate + 0.044715f * gate * gate * gate)));
    const float activated = gelu * (gated ? gate : 1.0f);
    const float expected =
        calibrate(std::nearbyint(activated * 128.0f) / 128.0f, 1.0f / 128.0f);
    for (size_t row = 0; row < rows; ++row) {
      for (size_t column = 0; column < output_width; ++column) {
        ASSERT_NEAR(output({row, column}), expected, 1e-5f);
      }
    }
  }
}

// When B is static, the packing is constant folded during
// ynn_optimize_subgraph: it has no dot loop nest to fuse into, so it runs
// with its own loops, which should still be parallelized.
TEST_F(LoopFusionTest, ConstantFoldedPackIsParallel) {
  const size_t K = 512, N = 4096;
  Tensor<float> b({K, N});
  b.fill(1.0f);

  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t out_id = 2;
  SubgraphBuilder subgraph(3);
  subgraph.AddInput(type_of<float>(), TensorShape(2), a_id)
      .AddTensor(b, b_id)
      .AddOutput(type_of<float>(), TensorShape(2), out_id)
      .AddDot(1, a_id, b_id, YNN_INVALID_VALUE_ID, out_id);

  // Constant folding runs the packing during MakeRuntime (in
  // ynn_optimize_subgraph); nothing else is invoked, so any tasks the
  // scheduler saw are the packing's parallel loops.
  MakeRuntime(subgraph.GetSubgraph());
// NOTE(vksnk): We skip WASM_SIMD128, because the default config doesn't enable
// threads so all of the loops become serial.
#if !defined(YNN_ARCH_WASM_SIMD128)
  EXPECT_GT(scheduler_.task_count(), 0);
#endif  // !YNN_ARCH_WASM_SIMD128
}

}  // namespace
}  // namespace ynn
