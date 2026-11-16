# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ADD_LIBRARY(ynnpack_composites STATIC
  attention_prefill.cc
  blockwise_linear.cc
  dense_linear.cc
  flash_attention.cc
  embedding.cc
  feed_forward.cc
  hnsw.cc
  hnsw_kernels.cc
  normalization.cc
  tensor_ops.cc
)

TARGET_INCLUDE_DIRECTORIES(ynnpack_composites PUBLIC .)
TARGET_LINK_LIBRARIES(ynnpack_composites PUBLIC
  ynnpack_base
  ynnpack_kernels_binary
  ynnpack_kernels_dot
  ynnpack_kernels_reduce
  ynnpack_kernels_transpose
  ynnpack_kernels_unary
)
IF(APPLE)
  FIND_LIBRARY(YNN_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  TARGET_COMPILE_DEFINITIONS(ynnpack_composites PRIVATE ACCELERATE_NEW_LAPACK)
  TARGET_LINK_LIBRARIES(ynnpack_composites PRIVATE ${YNN_ACCELERATE_FRAMEWORK})
ENDIF()

IF(YNNPACK_BUILD_TESTS)
  ADD_SUBDIRECTORY(test)
ENDIF()
