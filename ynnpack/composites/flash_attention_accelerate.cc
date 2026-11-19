// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>

#include <cstddef>

#include "ynnpack/composites/composites_internal.h"

namespace ynn::detail {

bool arch_sgemm_f32(bool transpose_b, size_t m, size_t n, size_t k, float alpha, const float* a, size_t lda,
                    const float* b, size_t ldb, float* c, size_t ldc) noexcept {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, transpose_b ? CblasTrans : CblasNoTrans, static_cast<int>(m),
                static_cast<int>(n), static_cast<int>(k), alpha, a, static_cast<int>(lda), b, static_cast<int>(ldb),
                0.0f, c, static_cast<int>(ldc));
    return true;
}

} // namespace ynn::detail
