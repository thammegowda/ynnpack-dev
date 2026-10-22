// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/base.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/to_string.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {

TEST(SubByteTypesRegression, TiledConvert2Bit) {
  // 131072 * 2 = 262144 elements logical.
  // Cache split threshold is 131072 elements, so we should get loop splits.
  std::vector<size_t> shapes_a = {4, 262144};
  std::vector<size_t> shapes_c = {4, 262144};

  SubgraphBuilder subgraph(3, 0);

  uint32_t a_id = 0;
  uint32_t b_id = 1;
  uint32_t output_id = 2;

  subgraph.AddInput(type_of<ynn::int2x4>(), shapes_a, a_id)
      .AddTensor(type_of<int8_t>(), shapes_a, b_id)
      .AddOutput(type_of<int32_t>(), 2, output_id);

  subgraph.AddConvert(a_id, type_of<int8_t>(), b_id, 0)
      .AddConvert(b_id, type_of<int32_t>(), output_id, 0);

  // We explicitly use the multi-threaded scheduler to enforce loop
  // partitioning.
  TestScheduler scheduler(3);
  Runtime runtime(subgraph.GetSubgraph(), &scheduler);
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  std::mt19937 rng(42);
  Tensor<ynn::int2x4> a(shapes_a);
  Tensor<int32_t> c(shapes_c);

  fill_random(a.data(), a.size(), rng, -2, 1);
  std::fill(c.begin(), c.end(), 0);

  runtime.SetupExternalTensor(a.data(), a_id)
      .SetupExternalTensor(c.data(), output_id)
      .ReshapeRuntime()
      .InvokeRuntime();

  ASSERT_EQ(runtime.Status(), ynn_status_success);

  // Reference comparison
  using A_info = type_info<ynn::int2x4>;
  for (size_t i = 0; i < 4; ++i) {
    const ynn::int2x4* a_i = address_of(a(i, 0));
    for (size_t j = 0; j < 262144; ++j) {
      int32_t expected_val = static_cast<int32_t>(A_info::get(a_i, j));
      ASSERT_EQ(c(i, j), expected_val) << "Mismatch at i=" << i << ", j=" << j;
    }
  }
}

void TestInt4TransposeDequantize(bool tensor_params) {
  constexpr size_t row_count = 4;
  constexpr size_t column_count = 4;
  auto value = [](size_t row, size_t column) -> int8_t {
    return static_cast<int8_t>(row * column_count + column) - 8;
  };
  std::vector<uint8_t> packed(row_count * column_count / 2, 0);
  for (size_t index = 0; index < row_count * column_count; ++index) {
    const uint8_t nibble =
        static_cast<uint8_t>(value(index / column_count, index % column_count));
    packed[index / 2] |= (nibble & 0xF) << (4 * (index % 2));
  }
  const std::vector<float> scales(tensor_params ? column_count * row_count : 1,
                                   1.0f);
  const std::vector<int32_t> zero_points(column_count * row_count, 0);
  ynn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(ynn_create_subgraph(1, 0, &subgraph), ynn_status_success);
  uint32_t output_id = 0;
  ASSERT_EQ(ynn_define_tensor(subgraph, ynn_type_fp32, 2, nullptr, nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
            ynn_status_success);
  const size_t dimensions[2] = {row_count, column_count};
  uint32_t int4_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(ynn_define_tensor(subgraph, ynn_type_int4, 2, dimensions,
                              packed.data(), 0, &int4_id),
            ynn_status_success);
  const int32_t permutation[2] = {1, 0};
  uint32_t transposed_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(ynn_define_static_transpose(subgraph, 2, permutation, int4_id,
                                        &transposed_id, 0),
            ynn_status_success);
  const size_t parameter_dimensions[2] = {column_count, row_count};
  const size_t parameter_rank = tensor_params ? 2 : 0;
  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(ynn_define_tensor(subgraph, ynn_type_fp32, parameter_rank,
                              parameter_dimensions, scales.data(),
                              YNN_VALUE_FLAG_COPY_DATA, &scale_id),
            ynn_status_success);
  uint32_t zero_point_id = YNN_INVALID_VALUE_ID;
  if (tensor_params) {
    ASSERT_EQ(ynn_define_tensor(subgraph, ynn_type_int32, parameter_rank,
                                parameter_dimensions, zero_points.data(),
                                YNN_VALUE_FLAG_COPY_DATA, &zero_point_id),
              ynn_status_success);
  }
  ASSERT_EQ(ynn_define_dequantize(subgraph, transposed_id, zero_point_id,
                                  scale_id, ynn_type_fp32, &output_id, 0),
            ynn_status_success);
  ASSERT_EQ(ynn_optimize_subgraph(subgraph, nullptr, 0), ynn_status_success);
  ynn_runtime_t runtime = nullptr;
  ASSERT_EQ(ynn_create_runtime(subgraph, nullptr, 0, &runtime),
            ynn_status_success);
  ASSERT_EQ(ynn_reshape_runtime(runtime), ynn_status_success);
  std::vector<float> output(column_count * row_count, -100.0f);
  ASSERT_EQ(ynn_set_external_value_data(runtime, output_id, output.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime), ynn_status_success);
  ynn_delete_runtime(runtime);
  ynn_delete_subgraph(subgraph);
  for (size_t column = 0; column < column_count; ++column) {
    for (size_t row = 0; row < row_count; ++row) {
      EXPECT_EQ(output[column * row_count + row],
                static_cast<float>(value(row, column)));
    }
  }
}

TEST(SubByteTypesRegression, Int4TransposeDequantizeScalarParams) {
  TestInt4TransposeDequantize(false);
}

TEST(SubByteTypesRegression, Int4TransposeDequantizeTensorParams) {
  TestInt4TransposeDequantize(true);
}

}  // namespace ynn
