# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

SET(YNN_COMPOSITES_SRCS
  attention_prefill.cc
  blockwise_linear.cc
  dense_linear.cc
  embedding.cc
  feed_forward.cc
  flash_attention.cc
  hnsw.cc
  hnsw_kernels.cc
  normalization.cc
  tensor_ops.cc
  vq_linear.cc
)
SET(YNN_COMPOSITES_DEFINES)

IF(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
  IF(YNNPACK_TARGET_PROCESSOR STREQUAL "x86_64" AND YNN_ENABLE_X86_AVX512)
    LIST(APPEND YNN_COMPOSITES_SRCS
      blockwise_linear_x86.cc
      embedding_x86.cc
      hnsw_kernels_x86.cc
      normalization_x86.cc
    )
    LIST(APPEND YNN_COMPOSITES_DEFINES YNN_COMPOSITES_X86=1)
  ELSEIF(YNN_ENABLE_ARM64_NEON)
    LIST(APPEND YNN_COMPOSITES_SRCS
      attention_prefill_neon.cc
      blockwise_linear_neon.cc
      dense_linear_neon.cc
      embedding_neon.cc
      flash_attention_neon.cc
      hnsw_kernels_neon.cc
      normalization_neon.cc
      tensor_ops_neon.cc
    )
    LIST(APPEND YNN_COMPOSITES_DEFINES YNN_COMPOSITES_NEON=1)
  ENDIF()
ENDIF()

IF(APPLE)
  LIST(APPEND YNN_COMPOSITES_SRCS flash_attention_accelerate.cc)
  LIST(APPEND YNN_COMPOSITES_DEFINES ACCELERATE_NEW_LAPACK YNN_COMPOSITES_ACCELERATE=1)
ENDIF()

ADD_LIBRARY(ynnpack_composites STATIC ${YNN_COMPOSITES_SRCS})

TARGET_INCLUDE_DIRECTORIES(ynnpack_composites PUBLIC .)
TARGET_COMPILE_DEFINITIONS(ynnpack_composites PRIVATE ${YNN_COMPOSITES_DEFINES})
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
  TARGET_LINK_LIBRARIES(ynnpack_composites PRIVATE ${YNN_ACCELERATE_FRAMEWORK})
ENDIF()

IF(YNNPACK_BUILD_TESTS)
  ADD_SUBDIRECTORY(test)
ENDIF()
