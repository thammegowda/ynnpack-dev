// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

using ynn::to_string;  // NOLINT(misc-unused-using-decls)

namespace ynn {
TEST(runtime, independent_external_bindings) {
  SubgraphBuilder builder(2);
  builder.AddInput(ynn_type_fp32, 1, 0)
    .AddOutput(ynn_type_fp32, 1, 1)
    .AddUnary(ynn_unary_negate, 0, 1);
  Runtime first(builder.GetSubgraph()), second(builder.GetSubgraph());
  ASSERT_EQ(first.Status(), ynn_status_success);
  ASSERT_EQ(second.Status(), ynn_status_success);
  std::vector<float> first_input{1, 2}, second_input{-3, -4, -5};
  std::vector<float> first_output(2), second_output(3);
  first.ReshapeExternalTensor({2}, first_input.data(), 0)
    .SetupExternalTensor(first_output.data(), 1).ReshapeRuntime();
  second.ReshapeExternalTensor({3}, second_input.data(), 0)
    .SetupExternalTensor(second_output.data(), 1).ReshapeRuntime();
  first.InvokeRuntime();
  second.InvokeRuntime();
  ASSERT_EQ(first.Status(), ynn_status_success);
  ASSERT_EQ(second.Status(), ynn_status_success);
  EXPECT_EQ(first.GetExternalTensorShape(1), std::vector<size_t>({2}));
  EXPECT_EQ(first_output, std::vector<float>({-1, -2}));
  EXPECT_EQ(second_output, std::vector<float>({3, 4, 5}));
}

TEST(runtime, retains_scratch_between_invocations) {
  constexpr uint32_t input_id = 0;
  constexpr uint32_t weight_id = 1;
  constexpr uint32_t output_id = 2;
  uint32_t product_id = YNN_INVALID_VALUE_ID;
  SubgraphBuilder builder(3);
  builder.AddInput(ynn_type_fp32, 2, input_id)
      .AddInput(ynn_type_fp32, {256, 256}, weight_id)
      .AddOutput(ynn_type_fp32, 2, output_id)
      .AddTensor(ynn_type_fp32, 2, product_id)
      .AddDot(1, input_id, weight_id, YNN_INVALID_VALUE_ID, product_id)
      .AddUnary(ynn_unary_square, product_id, output_id);

  size_t allocations = 0;
  size_t allocated_bytes = 0;
  size_t freed_bytes = 0;
  {
    Runtime runtime(builder.GetSubgraph(), nullptr, YNN_RUNTIME_FLAG_NO_SCHEDULE);
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    runtime.get()->eval_config.allocate = [&](size_t bytes, size_t alignment) {
      ++allocations;
      allocated_bytes += bytes;
      return slinky::allocate_bytes(bytes, alignment);
    };
    runtime.get()->eval_config.free = [&](void* pointer, size_t bytes) {
      freed_bytes += bytes;
      slinky::deallocate_bytes(pointer, bytes);
    };
    std::vector<float> weight(256 * 256, 1.0f / 256);
    runtime.SetupExternalTensor(weight.data(), weight_id);
    for (size_t rows : {256, 512, 128}) {
      std::vector<float> input(rows * 256, 2.0f);
      std::vector<float> output(rows * 256);
      runtime.ReshapeExternalTensor({rows, 256}, input.data(), input_id)
          .SetupExternalTensor(output.data(), output_id)
          .ReshapeRuntime()
          .InvokeRuntime();
      ASSERT_EQ(runtime.Status(), ynn_status_success);
      const auto after_first = allocations;
      EXPECT_GT(after_first, 0);
      EXPECT_GT(runtime.get()->eval_context.pool.retained_size(), 0);
      runtime.InvokeRuntime();
      ASSERT_EQ(runtime.Status(), ynn_status_success);
      EXPECT_EQ(allocations, after_first);
      for (float value : output) EXPECT_FLOAT_EQ(value, 4.0f);
    }
  }
  EXPECT_EQ(freed_bytes, allocated_bytes);
}

// By default WASM doesn't have any thread support.
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
TEST(runtime, dot_concurrency) {}
#else
TEST(runtime, dot_concurrency) {
  constexpr uint32_t a_id = 0;
  constexpr uint32_t b_id = 1;
  constexpr uint32_t c_id = 2;
  constexpr uint32_t init_zero = YNN_INVALID_VALUE_ID;

  TestScheduler scheduler(3);

  auto get_concurrency = [&](SubgraphBuilder& builder) -> int32_t {
    Runtime runtime(builder.GetSubgraph(), &scheduler);
    EXPECT_EQ(runtime.Status(), ynn_status_success);
    int32_t concurrency;
    EXPECT_EQ(runtime.Query(ynn_runtime_property_concurrency, &concurrency),
              ynn_status_success);
    return concurrency;
  };

  // We should be able to statically know this graph will not run a parallel
  // loop.
  SubgraphBuilder small(3);
  small.AddInput(type_of<float>(), {8, 8}, a_id);
  small.AddInput(type_of<float>(), {8, 8}, b_id);
  small.AddOutput(type_of<float>(), {8, 8}, c_id);
  small.AddDot(1, a_id, b_id, init_zero, c_id);
  // TODO(b/458542243): This doesn't actually work because we don't simplify
  // away these loops yet.
  // ASSERT_EQ(get_concurrency(small), 1);

  // We should be able to statically know this graph will run parallel loops.
  SubgraphBuilder big(3);
  big.AddInput(type_of<float>(), {800, 800}, a_id);
  big.AddInput(type_of<float>(), {800, 800}, b_id);
  big.AddOutput(type_of<float>(), {800, 800}, c_id);
  big.AddDot(1, a_id, b_id, init_zero, c_id);
  ASSERT_GT(get_concurrency(big), 1);

  // We don't know in this case, we might run a parallel loop if the input is
  // big enough.
  SubgraphBuilder dynamic(3);
  dynamic.AddInput(type_of<float>(), 2, a_id);
  dynamic.AddInput(type_of<float>(), 2, b_id);
  dynamic.AddOutput(type_of<float>(), 2, c_id);
  dynamic.AddDot(1, a_id, b_id, init_zero, c_id);
  ASSERT_GT(get_concurrency(dynamic), 1);
}
#endif
}  // namespace ynn
