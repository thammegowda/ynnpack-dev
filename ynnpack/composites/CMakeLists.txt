# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ADD_LIBRARY(ynnpack_composites STATIC
  attention_prefill.cc
  blockwise_linear.cc
  flash_attention.cc
  dot.cc
  dot_quantization.cc
  elementwise.cc
  embedding.cc
  feed_forward.cc
  normalization.cc
  pooling.cc
  reduction.cc
)

TARGET_INCLUDE_DIRECTORIES(ynnpack_composites PUBLIC .)
TARGET_LINK_LIBRARIES(ynnpack_composites PUBLIC
  ynnpack_base
  ynnpack_kernels_reduce
  ynnpack_kernels_unary
)

IF(YNNPACK_BUILD_TESTS)
  ADD_EXECUTABLE(ynnpack_composites_llm_test llm_composites_test.cc)
  TARGET_LINK_LIBRARIES(ynnpack_composites_llm_test
    ynnpack_composites
    ynnpack_subgraph
    ynnpack_base
    gtest
    gtest_main
  )
  ADD_TEST(NAME ynnpack_composites_llm_test COMMAND ynnpack_composites_llm_test)
ENDIF()
