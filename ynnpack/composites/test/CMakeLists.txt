# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ADD_EXECUTABLE(ynnpack_composites_llm_test llm_composites_test.cc)
TARGET_LINK_LIBRARIES(ynnpack_composites_llm_test ynnpack_composites gtest gtest_main)
ADD_TEST(NAME ynnpack_composites_llm_test COMMAND ynnpack_composites_llm_test)

ADD_EXECUTABLE(ynnpack_composites_hnsw_test hnsw_test.cc)
TARGET_LINK_LIBRARIES(ynnpack_composites_hnsw_test ynnpack_composites gtest gtest_main)
ADD_TEST(NAME ynnpack_composites_hnsw_test COMMAND ynnpack_composites_hnsw_test)

ADD_EXECUTABLE(ynnpack_composites_vq_linear_test vq_linear_test.cc)
TARGET_LINK_LIBRARIES(ynnpack_composites_vq_linear_test ynnpack_composites gtest gtest_main)
ADD_TEST(NAME ynnpack_composites_vq_linear_test COMMAND ynnpack_composites_vq_linear_test)
