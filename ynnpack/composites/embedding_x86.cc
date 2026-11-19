// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include "ynnpack/base/arch.h"
#include "ynnpack/composites/composites_internal.h"

namespace ynn::detail {
namespace {

YNN_X86_TARGET_AVX512 size_t quantize_symmetric_int8_avx512(size_t count, float scale, const float* input,
                                                            int8_t* output) noexcept {
    const __m512 vscale = _mm512_set1_ps(scale);
    const __m512i vmin = _mm512_set1_epi32(-128);
    const __m512i vmax = _mm512_set1_epi32(127);
    size_t index = 0;
    for (; index + 16 <= count; index += 16) {
        const __m512i q = _mm512_cvtps_epi32(_mm512_div_ps(_mm512_loadu_ps(input + index), vscale));
        const __m512i clamped = _mm512_min_epi32(_mm512_max_epi32(q, vmin), vmax);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + index), _mm512_cvtsepi32_epi8(clamped));
    }
    return index;
}

YNN_X86_TARGET_AVX512 size_t argmax_chunk_avx512(size_t count, const float* row_logits, size_t col_offset, float& best,
                                                 int32_t& token, bool& invalid) noexcept {
    __mmask16 bad_any = 0;
    const __m512i exp_mask = _mm512_set1_epi32(0x7F800000);
    const __m512i ninf_bits = _mm512_set1_epi32(static_cast<int32_t>(0xFF800000u));
    __m512 vbest = _mm512_set1_ps(best);
    size_t column = 0;
    for (; column + 16 <= count; column += 16) {
        const __m512 v = _mm512_loadu_ps(row_logits + column);
        const __m512i u = _mm512_castps_si512(v);
        const __mmask16 is_nonfinite = _mm512_cmpeq_epi32_mask(_mm512_and_si512(u, exp_mask), exp_mask);
        const __mmask16 not_ninf = _mm512_cmpneq_epi32_mask(u, ninf_bits);
        bad_any |= static_cast<__mmask16>(is_nonfinite & not_ninf);
        if (_mm512_cmp_ps_mask(v, vbest, _CMP_GT_OQ) != 0) {
            for (size_t i = 0; i < 16; ++i) {
                const float value = row_logits[column + i];
                if (value > best) {
                    best = value;
                    token = static_cast<int32_t>(col_offset + column + i);
                }
            }
            vbest = _mm512_set1_ps(best);
        }
    }
    if (bad_any != 0) invalid = true;
    return column;
}

} // namespace

size_t arch_quantize_symmetric_int8_f32(size_t count, float scale, const float* input, int8_t* output) noexcept {
    if (is_arch_supported(arch_flag::avx512)) {
        return quantize_symmetric_int8_avx512(count, scale, input, output);
    }
    return 0;
}

size_t arch_argmax_chunk_f32(size_t count, const float* row_logits, size_t col_offset, float& best, int32_t& token,
                             bool& invalid) noexcept {
    if (is_arch_supported(arch_flag::avx512)) {
        return argmax_chunk_avx512(count, row_logits, col_offset, best, token, invalid);
    }
    return 0;
}

} // namespace ynn::detail
