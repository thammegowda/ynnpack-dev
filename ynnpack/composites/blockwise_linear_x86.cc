// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/half.h"
#include "ynnpack/composites/composites_internal.h"

namespace ynn::detail {
namespace {

inline bool has_x86_vnni() noexcept {
    return is_arch_supported(arch_flag::avx512 | arch_flag::avx512vnni | arch_flag::fma3 | arch_flag::f16c);
}

inline bool has_x86_amx_int8() noexcept {
    return is_arch_supported(arch_flag::avx512 | arch_flag::avx512vnni | arch_flag::fma3 | arch_flag::f16c |
                             arch_flag::amxint8);
}

struct alignas(64) x86_amx_tile_config {
    uint8_t palette_id = 0;
    uint8_t start_row = 0;
    uint8_t reserved_0[14] = {};
    uint16_t colsb[8] = {};
    uint16_t reserved_1[8] = {};
    uint8_t rows[8] = {};
    uint8_t reserved_2[8] = {};
};

YNN_X86_TARGET_VNNI inline void store_row32_x86(float* dst, __m512 acc_lo, __m512 acc_hi, size_t n_valid,
                                                float output_scale) noexcept {
    if (output_scale > 0.0f) {
        const __m512 vscale = _mm512_set1_ps(output_scale);
        const __m512 vmin = _mm512_set1_ps(-128.0f);
        const __m512 vmax = _mm512_set1_ps(127.0f);
        acc_lo = _mm512_mul_ps(
            _mm512_min_ps(_mm512_max_ps(_mm512_roundscale_ps(_mm512_div_ps(acc_lo, vscale),
                                                             _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
                                        vmin),
                          vmax),
            vscale);
        acc_hi = _mm512_mul_ps(
            _mm512_min_ps(_mm512_max_ps(_mm512_roundscale_ps(_mm512_div_ps(acc_hi, vscale),
                                                             _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
                                        vmin),
                          vmax),
            vscale);
    }
    if (n_valid >= 32) {
        _mm512_storeu_ps(dst + 0, acc_lo);
        _mm512_storeu_ps(dst + 16, acc_hi);
    } else if (n_valid > 16) {
        _mm512_storeu_ps(dst + 0, acc_lo);
        const __mmask16 mask = static_cast<__mmask16>((1u << (n_valid - 16)) - 1u);
        _mm512_mask_storeu_ps(dst + 16, mask, acc_hi);
    } else if (n_valid > 0) {
        const __mmask16 mask = n_valid == 16 ? 0xFFFFu : static_cast<__mmask16>((1u << n_valid) - 1u);
        _mm512_mask_storeu_ps(dst + 0, mask, acc_lo);
    }
}

YNN_X86_TARGET_VNNI void eval_i8_tile_32x1_x86(size_t B, size_t block_size, const uint8_t* p, const uint8_t* qp,
                                               const float* a_scale_base, const float* a_zp_base, float* y_base,
                                               size_t n_valid, float output_scale) noexcept {
    __m512 acc_lo = _mm512_setzero_ps();
    __m512 acc_hi = _mm512_setzero_ps();
    for (size_t b = 0; b < B; ++b) {
        __m512i d0_lo = _mm512_setzero_si512(), d0_hi = _mm512_setzero_si512();
        __m512i d1_lo = _mm512_setzero_si512(), d1_hi = _mm512_setzero_si512();
        __m512i d2_lo = _mm512_setzero_si512(), d2_hi = _mm512_setzero_si512();
        __m512i d3_lo = _mm512_setzero_si512(), d3_hi = _mm512_setzero_si512();
        size_t k = 0;
        for (; k + 16 <= block_size; k += 16, p += 512, qp += 16) {
            int32_t q0, q1, q2, q3;
            std::memcpy(&q0, qp + 0, sizeof(int32_t));
            std::memcpy(&q1, qp + 4, sizeof(int32_t));
            std::memcpy(&q2, qp + 8, sizeof(int32_t));
            std::memcpy(&q3, qp + 12, sizeof(int32_t));
            const __m512i qa0 = _mm512_set1_epi32(q0);
            const __m512i qa1 = _mm512_set1_epi32(q1);
            const __m512i qa2 = _mm512_set1_epi32(q2);
            const __m512i qa3 = _mm512_set1_epi32(q3);
            d0_lo = _mm512_dpbusd_epi32(d0_lo, qa0, _mm512_loadu_si512(p + 0));
            d0_hi = _mm512_dpbusd_epi32(d0_hi, qa0, _mm512_loadu_si512(p + 64));
            d1_lo = _mm512_dpbusd_epi32(d1_lo, qa1, _mm512_loadu_si512(p + 128));
            d1_hi = _mm512_dpbusd_epi32(d1_hi, qa1, _mm512_loadu_si512(p + 192));
            d2_lo = _mm512_dpbusd_epi32(d2_lo, qa2, _mm512_loadu_si512(p + 256));
            d2_hi = _mm512_dpbusd_epi32(d2_hi, qa2, _mm512_loadu_si512(p + 320));
            d3_lo = _mm512_dpbusd_epi32(d3_lo, qa3, _mm512_loadu_si512(p + 384));
            d3_hi = _mm512_dpbusd_epi32(d3_hi, qa3, _mm512_loadu_si512(p + 448));
        }
        __m512i d_lo = _mm512_add_epi32(_mm512_add_epi32(d0_lo, d1_lo), _mm512_add_epi32(d2_lo, d3_lo));
        __m512i d_hi = _mm512_add_epi32(_mm512_add_epi32(d0_hi, d1_hi), _mm512_add_epi32(d2_hi, d3_hi));
        for (; k < block_size; k += 4, p += 128, qp += 4) {
            int32_t q0 = 0;
            std::memcpy(&q0, qp, sizeof(int32_t));
            const __m512i qa0 = _mm512_set1_epi32(q0);
            d_lo = _mm512_dpbusd_epi32(d_lo, qa0, _mm512_loadu_si512(p + 0));
            d_hi = _mm512_dpbusd_epi32(d_hi, qa0, _mm512_loadu_si512(p + 64));
        }
        const __m512 ws_lo = _mm512_loadu_ps(reinterpret_cast<const float*>(p + 0));
        const __m512 ws_hi = _mm512_loadu_ps(reinterpret_cast<const float*>(p + 64));
        p += 128;
        const __m512i wsum_lo = _mm512_loadu_si512(p + 0);
        const __m512i wsum_hi = _mm512_loadu_si512(p + 64);
        p += 128;
        const int32_t zp_u8 = a_zp_base != nullptr ? static_cast<int32_t>(a_zp_base[b]) : 128;
        const __m512i vzp = _mm512_set1_epi32(zp_u8);
        const __m512 vas = _mm512_set1_ps(a_scale_base[b]);
        const __m512i corr_lo = _mm512_sub_epi32(d_lo, _mm512_mullo_epi32(wsum_lo, vzp));
        const __m512i corr_hi = _mm512_sub_epi32(d_hi, _mm512_mullo_epi32(wsum_hi, vzp));
        acc_lo = _mm512_fmadd_ps(_mm512_cvtepi32_ps(corr_lo), _mm512_mul_ps(ws_lo, vas), acc_lo);
        acc_hi = _mm512_fmadd_ps(_mm512_cvtepi32_ps(corr_hi), _mm512_mul_ps(ws_hi, vas), acc_hi);
    }
    store_row32_x86(y_base, acc_lo, acc_hi, n_valid, output_scale);
}

YNN_X86_TARGET_VNNI void eval_i8_tile_64x1_x86(size_t B, size_t block_size, const uint8_t* p0, const uint8_t* p1,
                                               const uint8_t* qp, const float* a_scale_base, const float* a_zp_base,
                                               float* y_base, float output_scale) noexcept {
    __m512 acc0_lo = _mm512_setzero_ps(), acc0_hi = _mm512_setzero_ps();
    __m512 acc1_lo = _mm512_setzero_ps(), acc1_hi = _mm512_setzero_ps();
    for (size_t b = 0; b < B; ++b) {
        __m512i t0_lo0 = _mm512_setzero_si512(), t0_hi0 = _mm512_setzero_si512();
        __m512i t1_lo0 = _mm512_setzero_si512(), t1_hi0 = _mm512_setzero_si512();
        __m512i t0_lo1 = _mm512_setzero_si512(), t0_hi1 = _mm512_setzero_si512();
        __m512i t1_lo1 = _mm512_setzero_si512(), t1_hi1 = _mm512_setzero_si512();
        size_t k = 0;
        for (; k + 16 <= block_size; k += 16, p0 += 512, p1 += 512, qp += 16) {
            int32_t q0, q1, q2, q3;
            std::memcpy(&q0, qp + 0, sizeof(int32_t));
            std::memcpy(&q1, qp + 4, sizeof(int32_t));
            std::memcpy(&q2, qp + 8, sizeof(int32_t));
            std::memcpy(&q3, qp + 12, sizeof(int32_t));
            const __m512i qa0 = _mm512_set1_epi32(q0);
            const __m512i qa1 = _mm512_set1_epi32(q1);
            const __m512i qa2 = _mm512_set1_epi32(q2);
            const __m512i qa3 = _mm512_set1_epi32(q3);
            t0_lo0 = _mm512_dpbusd_epi32(t0_lo0, qa0, _mm512_loadu_si512(p0 + 0));
            t0_hi0 = _mm512_dpbusd_epi32(t0_hi0, qa0, _mm512_loadu_si512(p0 + 64));
            t1_lo0 = _mm512_dpbusd_epi32(t1_lo0, qa0, _mm512_loadu_si512(p1 + 0));
            t1_hi0 = _mm512_dpbusd_epi32(t1_hi0, qa0, _mm512_loadu_si512(p1 + 64));
            t0_lo1 = _mm512_dpbusd_epi32(t0_lo1, qa1, _mm512_loadu_si512(p0 + 128));
            t0_hi1 = _mm512_dpbusd_epi32(t0_hi1, qa1, _mm512_loadu_si512(p0 + 192));
            t1_lo1 = _mm512_dpbusd_epi32(t1_lo1, qa1, _mm512_loadu_si512(p1 + 128));
            t1_hi1 = _mm512_dpbusd_epi32(t1_hi1, qa1, _mm512_loadu_si512(p1 + 192));
            t0_lo0 = _mm512_dpbusd_epi32(t0_lo0, qa2, _mm512_loadu_si512(p0 + 256));
            t0_hi0 = _mm512_dpbusd_epi32(t0_hi0, qa2, _mm512_loadu_si512(p0 + 320));
            t1_lo0 = _mm512_dpbusd_epi32(t1_lo0, qa2, _mm512_loadu_si512(p1 + 256));
            t1_hi0 = _mm512_dpbusd_epi32(t1_hi0, qa2, _mm512_loadu_si512(p1 + 320));
            t0_lo1 = _mm512_dpbusd_epi32(t0_lo1, qa3, _mm512_loadu_si512(p0 + 384));
            t0_hi1 = _mm512_dpbusd_epi32(t0_hi1, qa3, _mm512_loadu_si512(p0 + 448));
            t1_lo1 = _mm512_dpbusd_epi32(t1_lo1, qa3, _mm512_loadu_si512(p1 + 384));
            t1_hi1 = _mm512_dpbusd_epi32(t1_hi1, qa3, _mm512_loadu_si512(p1 + 448));
        }
        __m512i d0_lo = _mm512_add_epi32(t0_lo0, t0_lo1), d0_hi = _mm512_add_epi32(t0_hi0, t0_hi1);
        __m512i d1_lo = _mm512_add_epi32(t1_lo0, t1_lo1), d1_hi = _mm512_add_epi32(t1_hi0, t1_hi1);
        for (; k < block_size; k += 4, p0 += 128, p1 += 128, qp += 4) {
            int32_t q0 = 0;
            std::memcpy(&q0, qp, sizeof(int32_t));
            const __m512i qa0 = _mm512_set1_epi32(q0);
            d0_lo = _mm512_dpbusd_epi32(d0_lo, qa0, _mm512_loadu_si512(p0 + 0));
            d0_hi = _mm512_dpbusd_epi32(d0_hi, qa0, _mm512_loadu_si512(p0 + 64));
            d1_lo = _mm512_dpbusd_epi32(d1_lo, qa0, _mm512_loadu_si512(p1 + 0));
            d1_hi = _mm512_dpbusd_epi32(d1_hi, qa0, _mm512_loadu_si512(p1 + 64));
        }
        const int32_t zp_u8 = a_zp_base != nullptr ? static_cast<int32_t>(a_zp_base[b]) : 128;
        const __m512i vzp = _mm512_set1_epi32(zp_u8);
        const __m512 vas = _mm512_set1_ps(a_scale_base[b]);

        const __m512 ws0_lo = _mm512_loadu_ps(reinterpret_cast<const float*>(p0 + 0));
        const __m512 ws0_hi = _mm512_loadu_ps(reinterpret_cast<const float*>(p0 + 64));
        const __m512i wsum0_lo = _mm512_loadu_si512(p0 + 128);
        const __m512i wsum0_hi = _mm512_loadu_si512(p0 + 192);
        p0 += 256;

        const __m512 ws1_lo = _mm512_loadu_ps(reinterpret_cast<const float*>(p1 + 0));
        const __m512 ws1_hi = _mm512_loadu_ps(reinterpret_cast<const float*>(p1 + 64));
        const __m512i wsum1_lo = _mm512_loadu_si512(p1 + 128);
        const __m512i wsum1_hi = _mm512_loadu_si512(p1 + 192);
        p1 += 256;

        acc0_lo = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(d0_lo, _mm512_mullo_epi32(wsum0_lo, vzp))),
                                  _mm512_mul_ps(ws0_lo, vas), acc0_lo);
        acc0_hi = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(d0_hi, _mm512_mullo_epi32(wsum0_hi, vzp))),
                                  _mm512_mul_ps(ws0_hi, vas), acc0_hi);
        acc1_lo = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(d1_lo, _mm512_mullo_epi32(wsum1_lo, vzp))),
                                  _mm512_mul_ps(ws1_lo, vas), acc1_lo);
        acc1_hi = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(d1_hi, _mm512_mullo_epi32(wsum1_hi, vzp))),
                                  _mm512_mul_ps(ws1_hi, vas), acc1_hi);
    }
    store_row32_x86(y_base + 0, acc0_lo, acc0_hi, 32, output_scale);
    store_row32_x86(y_base + 32, acc1_lo, acc1_hi, 32, output_scale);
}

YNN_X86_TARGET_VNNI void eval_i8_tile_128x1_x86(size_t B, size_t block_size, const uint8_t* p0, const uint8_t* p1,
                                                const uint8_t* p2, const uint8_t* p3, const uint8_t* qp,
                                                const float* a_scale_base, const float* a_zp_base, float* y_base,
                                                float output_scale) noexcept {
    __m512 acc0_lo = _mm512_setzero_ps(), acc0_hi = _mm512_setzero_ps();
    __m512 acc1_lo = _mm512_setzero_ps(), acc1_hi = _mm512_setzero_ps();
    __m512 acc2_lo = _mm512_setzero_ps(), acc2_hi = _mm512_setzero_ps();
    __m512 acc3_lo = _mm512_setzero_ps(), acc3_hi = _mm512_setzero_ps();
    for (size_t b = 0; b < B; ++b) {
        __m512i d0_lo = _mm512_setzero_si512(), d0_hi = _mm512_setzero_si512();
        __m512i d1_lo = _mm512_setzero_si512(), d1_hi = _mm512_setzero_si512();
        __m512i d2_lo = _mm512_setzero_si512(), d2_hi = _mm512_setzero_si512();
        __m512i d3_lo = _mm512_setzero_si512(), d3_hi = _mm512_setzero_si512();
        size_t k = 0;
        for (; k + 8 <= block_size; k += 8, p0 += 256, p1 += 256, p2 += 256, p3 += 256, qp += 8) {
            int32_t q0, q1;
            std::memcpy(&q0, qp + 0, sizeof(int32_t));
            std::memcpy(&q1, qp + 4, sizeof(int32_t));
            const __m512i qa0 = _mm512_set1_epi32(q0);
            const __m512i qa1 = _mm512_set1_epi32(q1);
            d0_lo = _mm512_dpbusd_epi32(d0_lo, qa0, _mm512_loadu_si512(p0 + 0));
            d0_hi = _mm512_dpbusd_epi32(d0_hi, qa0, _mm512_loadu_si512(p0 + 64));
            d1_lo = _mm512_dpbusd_epi32(d1_lo, qa0, _mm512_loadu_si512(p1 + 0));
            d1_hi = _mm512_dpbusd_epi32(d1_hi, qa0, _mm512_loadu_si512(p1 + 64));
            d2_lo = _mm512_dpbusd_epi32(d2_lo, qa0, _mm512_loadu_si512(p2 + 0));
            d2_hi = _mm512_dpbusd_epi32(d2_hi, qa0, _mm512_loadu_si512(p2 + 64));
            d3_lo = _mm512_dpbusd_epi32(d3_lo, qa0, _mm512_loadu_si512(p3 + 0));
            d3_hi = _mm512_dpbusd_epi32(d3_hi, qa0, _mm512_loadu_si512(p3 + 64));

            d0_lo = _mm512_dpbusd_epi32(d0_lo, qa1, _mm512_loadu_si512(p0 + 128));
            d0_hi = _mm512_dpbusd_epi32(d0_hi, qa1, _mm512_loadu_si512(p0 + 192));
            d1_lo = _mm512_dpbusd_epi32(d1_lo, qa1, _mm512_loadu_si512(p1 + 128));
            d1_hi = _mm512_dpbusd_epi32(d1_hi, qa1, _mm512_loadu_si512(p1 + 192));
            d2_lo = _mm512_dpbusd_epi32(d2_lo, qa1, _mm512_loadu_si512(p2 + 128));
            d2_hi = _mm512_dpbusd_epi32(d2_hi, qa1, _mm512_loadu_si512(p2 + 192));
            d3_lo = _mm512_dpbusd_epi32(d3_lo, qa1, _mm512_loadu_si512(p3 + 128));
            d3_hi = _mm512_dpbusd_epi32(d3_hi, qa1, _mm512_loadu_si512(p3 + 192));
        }
        for (; k < block_size; k += 4, p0 += 128, p1 += 128, p2 += 128, p3 += 128, qp += 4) {
            int32_t q0 = 0;
            std::memcpy(&q0, qp, sizeof(int32_t));
            const __m512i qa0 = _mm512_set1_epi32(q0);
            d0_lo = _mm512_dpbusd_epi32(d0_lo, qa0, _mm512_loadu_si512(p0 + 0));
            d0_hi = _mm512_dpbusd_epi32(d0_hi, qa0, _mm512_loadu_si512(p0 + 64));
            d1_lo = _mm512_dpbusd_epi32(d1_lo, qa0, _mm512_loadu_si512(p1 + 0));
            d1_hi = _mm512_dpbusd_epi32(d1_hi, qa0, _mm512_loadu_si512(p1 + 64));
            d2_lo = _mm512_dpbusd_epi32(d2_lo, qa0, _mm512_loadu_si512(p2 + 0));
            d2_hi = _mm512_dpbusd_epi32(d2_hi, qa0, _mm512_loadu_si512(p2 + 64));
            d3_lo = _mm512_dpbusd_epi32(d3_lo, qa0, _mm512_loadu_si512(p3 + 0));
            d3_hi = _mm512_dpbusd_epi32(d3_hi, qa0, _mm512_loadu_si512(p3 + 64));
        }
        const int32_t zp_u8 = a_zp_base != nullptr ? static_cast<int32_t>(a_zp_base[b]) : 128;
        const __m512i vzp = _mm512_set1_epi32(zp_u8);
        const __m512 vas = _mm512_set1_ps(a_scale_base[b]);

        acc0_lo = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_sub_epi32(d0_lo, _mm512_mullo_epi32(_mm512_loadu_si512(p0 + 128), vzp))),
            _mm512_mul_ps(_mm512_loadu_ps(reinterpret_cast<const float*>(p0 + 0)), vas), acc0_lo);
        acc0_hi = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_sub_epi32(d0_hi, _mm512_mullo_epi32(_mm512_loadu_si512(p0 + 192), vzp))),
            _mm512_mul_ps(_mm512_loadu_ps(reinterpret_cast<const float*>(p0 + 64)), vas), acc0_hi);
        p0 += 256;

        acc1_lo = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_sub_epi32(d1_lo, _mm512_mullo_epi32(_mm512_loadu_si512(p1 + 128), vzp))),
            _mm512_mul_ps(_mm512_loadu_ps(reinterpret_cast<const float*>(p1 + 0)), vas), acc1_lo);
        acc1_hi = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_sub_epi32(d1_hi, _mm512_mullo_epi32(_mm512_loadu_si512(p1 + 192), vzp))),
            _mm512_mul_ps(_mm512_loadu_ps(reinterpret_cast<const float*>(p1 + 64)), vas), acc1_hi);
        p1 += 256;

        acc2_lo = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_sub_epi32(d2_lo, _mm512_mullo_epi32(_mm512_loadu_si512(p2 + 128), vzp))),
            _mm512_mul_ps(_mm512_loadu_ps(reinterpret_cast<const float*>(p2 + 0)), vas), acc2_lo);
        acc2_hi = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_sub_epi32(d2_hi, _mm512_mullo_epi32(_mm512_loadu_si512(p2 + 192), vzp))),
            _mm512_mul_ps(_mm512_loadu_ps(reinterpret_cast<const float*>(p2 + 64)), vas), acc2_hi);
        p2 += 256;

        acc3_lo = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_sub_epi32(d3_lo, _mm512_mullo_epi32(_mm512_loadu_si512(p3 + 128), vzp))),
            _mm512_mul_ps(_mm512_loadu_ps(reinterpret_cast<const float*>(p3 + 0)), vas), acc3_lo);
        acc3_hi = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_sub_epi32(d3_hi, _mm512_mullo_epi32(_mm512_loadu_si512(p3 + 192), vzp))),
            _mm512_mul_ps(_mm512_loadu_ps(reinterpret_cast<const float*>(p3 + 64)), vas), acc3_hi);
        p3 += 256;
    }
    store_row32_x86(y_base + 0, acc0_lo, acc0_hi, 32, output_scale);
    store_row32_x86(y_base + 32, acc1_lo, acc1_hi, 32, output_scale);
    store_row32_x86(y_base + 64, acc2_lo, acc2_hi, 32, output_scale);
    store_row32_x86(y_base + 96, acc3_lo, acc3_hi, 32, output_scale);
}

YNN_X86_TARGET_AMX void dispatch_i8_tiles_amx(size_t M, size_t N, size_t B, size_t a_row_stride, size_t block_size,
                                              float output_scale, const uint8_t* packed_ptr, const uint8_t* act_u8,
                                              const float* act_scale, const float* act_zp, float* y, size_t y_stride,
                                              size_t t_begin, size_t t_end) noexcept {
    x86_amx_tile_config cfg{};
    cfg.palette_id = 1;
    for (int t = 0; t < 8; ++t) {
        cfg.rows[t] = 16;
        cfg.colsb[t] = 64;
    }
    _tile_loadconfig(&cfg);

    const size_t block_bytes = block_size * 32 + 256;
    const size_t tile_bytes = B * block_bytes;
    const size_t k_amx = block_size & ~size_t{63};
    constexpr size_t M_TILE = 32;
    alignas(64) float acc_buf[M_TILE][32];
    alignas(64) int32_t d_buf[M_TILE][32];
    alignas(64) uint8_t a_tail[16][64];

    for (size_t tile = t_begin; tile < t_end; ++tile) {
        const size_t n0 = tile * 32;
        const size_t n_valid = std::min<size_t>(32, N - n0);
        const size_t col_off = (y_stride == 32 && t_end == t_begin + 1) ? 0 : n0;
        const uint8_t* tile_ptr = packed_ptr + tile * tile_bytes;

        for (size_t m0 = 0; m0 < M; m0 += M_TILE) {
            const size_t m_count = std::min<size_t>(M_TILE, M - m0);
            std::memset(acc_buf, 0, m_count * 32 * sizeof(float));
            const uint8_t* p_block = tile_ptr;
            for (size_t b = 0; b < B; ++b, p_block += block_bytes) {
                const uint8_t* a_base = act_u8 + m0 * a_row_stride + b * block_size;
                _tile_zero(0);
                _tile_zero(1);
                if (m_count > 16) {
                    _tile_zero(2);
                    _tile_zero(3);
                }
                const size_t tail_rows = m_count < 16 ? m_count : (m_count < 32 ? m_count - 16 : 0);
                if (tail_rows > 0) {
                    std::memset(a_tail[tail_rows], 0, (16 - tail_rows) * 64);
                }
                for (size_t k0 = 0; k0 < k_amx; k0 += 64) {
                    const uint8_t* p_k = p_block + k0 * 32;
                    if (m_count >= 16) {
                        _tile_loadd(4, a_base + k0, a_row_stride);
                    } else {
                        for (size_t r = 0; r < m_count; ++r) {
                            _mm512_store_si512(a_tail[r], _mm512_loadu_si512(a_base + r * a_row_stride + k0));
                        }
                        _tile_loadd(4, a_tail, 64);
                    }
                    _tile_loadd(6, p_k + 0, 128);
                    YNN_X86_AMX_DPBUSD(0, 4, 6);
                    _tile_loadd(7, p_k + 64, 128);
                    YNN_X86_AMX_DPBUSD(1, 4, 7);
                    if (m_count > 16) {
                        if (m_count == 32) {
                            _tile_loadd(5, a_base + 16 * a_row_stride + k0, a_row_stride);
                        } else {
                            for (size_t r = 0; r < m_count - 16; ++r) {
                                _mm512_store_si512(a_tail[r],
                                                   _mm512_loadu_si512(a_base + (16 + r) * a_row_stride + k0));
                            }
                            _tile_loadd(5, a_tail, 64);
                        }
                        YNN_X86_AMX_DPBUSD(2, 5, 6);
                        YNN_X86_AMX_DPBUSD(3, 5, 7);
                    }
                }
                _tile_stored(0, &d_buf[0][0], 128);
                _tile_stored(1, &d_buf[0][16], 128);
                if (m_count > 16) {
                    _tile_stored(2, &d_buf[16][0], 128);
                    _tile_stored(3, &d_buf[16][16], 128);
                }

                if (k_amx < block_size) {
                    for (size_t m = 0; m < m_count; ++m) {
                        __m512i d_lo = _mm512_load_si512(&d_buf[m][0]);
                        __m512i d_hi = _mm512_load_si512(&d_buf[m][16]);
                        const uint8_t* p_k = p_block + k_amx * 32;
                        const uint8_t* qp = a_base + m * a_row_stride + k_amx;
                        for (size_t k = k_amx; k < block_size; k += 4, p_k += 128, qp += 4) {
                            int32_t q0 = 0;
                            std::memcpy(&q0, qp, sizeof(int32_t));
                            const __m512i qa = _mm512_set1_epi32(q0);
                            d_lo = _mm512_dpbusd_epi32(d_lo, qa, _mm512_loadu_si512(p_k + 0));
                            d_hi = _mm512_dpbusd_epi32(d_hi, qa, _mm512_loadu_si512(p_k + 64));
                        }
                        _mm512_store_si512(&d_buf[m][0], d_lo);
                        _mm512_store_si512(&d_buf[m][16], d_hi);
                    }
                }

                const auto* sp = reinterpret_cast<const float*>(p_block + block_size * 32);
                const __m512 ws_lo = _mm512_loadu_ps(sp + 0);
                const __m512 ws_hi = _mm512_loadu_ps(sp + 16);
                const __m512i wsum_lo = _mm512_loadu_si512(p_block + block_size * 32 + 128);
                const __m512i wsum_hi = _mm512_loadu_si512(p_block + block_size * 32 + 192);
                for (size_t m = 0; m < m_count; ++m) {
                    const float as = act_scale[(m0 + m) * B + b];
                    const int32_t zp_u8 = act_zp != nullptr ? static_cast<int32_t>(act_zp[(m0 + m) * B + b]) : 128;
                    const __m512i vzp = _mm512_set1_epi32(zp_u8);
                    const __m512 vas = _mm512_set1_ps(as);
                    const __m512i corr_lo =
                        _mm512_sub_epi32(_mm512_load_si512(&d_buf[m][0]), _mm512_mullo_epi32(wsum_lo, vzp));
                    const __m512i corr_hi =
                        _mm512_sub_epi32(_mm512_load_si512(&d_buf[m][16]), _mm512_mullo_epi32(wsum_hi, vzp));
                    _mm512_store_ps(acc_buf[m] + 0,
                                    _mm512_fmadd_ps(_mm512_cvtepi32_ps(corr_lo), _mm512_mul_ps(ws_lo, vas),
                                                    _mm512_load_ps(acc_buf[m] + 0)));
                    _mm512_store_ps(acc_buf[m] + 16,
                                    _mm512_fmadd_ps(_mm512_cvtepi32_ps(corr_hi), _mm512_mul_ps(ws_hi, vas),
                                                    _mm512_load_ps(acc_buf[m] + 16)));
                }
            }
            for (size_t m = 0; m < m_count; ++m) {
                store_row32_x86(y + (m0 + m) * y_stride + col_off, _mm512_load_ps(acc_buf[m] + 0),
                                _mm512_load_ps(acc_buf[m] + 16), n_valid, output_scale);
            }
        }
    }
    _tile_release();
}

YNN_X86_TARGET_VNNI void dispatch_i8_tiles_vnni(size_t M, size_t N, size_t B, size_t a_row_stride, size_t block_size,
                                                float output_scale, const uint8_t* packed_ptr, const uint8_t* act_u8,
                                                const float* act_scale, const float* act_zp, float* y, size_t y_stride,
                                                size_t t_begin, size_t t_end) noexcept {
    const size_t block_bytes = block_size * 32 + 256;
    const size_t tile_bytes = B * block_bytes;

    if (M == 1) {
        size_t tile = t_begin;
        if (!(y_stride == 32 && t_end == t_begin + 1)) {
            for (; tile + 4 <= t_end && (tile + 4) * 32 <= N; tile += 4) {
                eval_i8_tile_128x1_x86(B, block_size, packed_ptr + tile * tile_bytes,
                                       packed_ptr + (tile + 1) * tile_bytes, packed_ptr + (tile + 2) * tile_bytes,
                                       packed_ptr + (tile + 3) * tile_bytes, act_u8, act_scale, act_zp, y + tile * 32,
                                       output_scale);
            }
            for (; tile + 2 <= t_end && (tile + 2) * 32 <= N; tile += 2) {
                eval_i8_tile_64x1_x86(B, block_size, packed_ptr + tile * tile_bytes,
                                      packed_ptr + (tile + 1) * tile_bytes, act_u8, act_scale, act_zp, y + tile * 32,
                                      output_scale);
            }
        }
        for (; tile < t_end; ++tile) {
            const size_t n0 = tile * 32;
            const size_t n_valid = std::min<size_t>(32, N - n0);
            const size_t col_off = (y_stride == 32 && t_end == t_begin + 1) ? 0 : n0;
            eval_i8_tile_32x1_x86(B, block_size, packed_ptr + tile * tile_bytes, act_u8, act_scale, act_zp, y + col_off,
                                  n_valid, output_scale);
        }
        return;
    }

    constexpr size_t M_TILE = 32;
    alignas(64) float acc_buf[M_TILE][32];
    for (size_t tile = t_begin; tile < t_end; ++tile) {
        const size_t n0 = tile * 32;
        const size_t n_valid = std::min<size_t>(32, N - n0);
        const size_t col_off = (y_stride == 32 && t_end == t_begin + 1) ? 0 : n0;
        const uint8_t* tile_ptr = packed_ptr + tile * tile_bytes;

        for (size_t m0 = 0; m0 < M; m0 += M_TILE) {
            const size_t m_count = std::min<size_t>(M_TILE, M - m0);
            std::memset(acc_buf, 0, m_count * 32 * sizeof(float));
            const uint8_t* p_block = tile_ptr;
            for (size_t b = 0; b < B; ++b, p_block += block_bytes) {
                const auto* sp = reinterpret_cast<const float*>(p_block + block_size * 32);
                const __m512 ws_lo = _mm512_loadu_ps(sp + 0);
                const __m512 ws_hi = _mm512_loadu_ps(sp + 16);
                const __m512i wsum_lo = _mm512_loadu_si512(p_block + block_size * 32 + 128);
                const __m512i wsum_hi = _mm512_loadu_si512(p_block + block_size * 32 + 192);

                size_t m = 0;
                for (; m + 4 <= m_count; m += 4) {
                    __m512i d0_lo = _mm512_setzero_si512(), d0_hi = _mm512_setzero_si512();
                    __m512i d1_lo = _mm512_setzero_si512(), d1_hi = _mm512_setzero_si512();
                    __m512i d2_lo = _mm512_setzero_si512(), d2_hi = _mm512_setzero_si512();
                    __m512i d3_lo = _mm512_setzero_si512(), d3_hi = _mm512_setzero_si512();
                    const uint8_t* p_k = p_block;
                    const uint8_t* q0 = act_u8 + (m0 + m + 0) * a_row_stride + b * block_size;
                    const uint8_t* q1 = act_u8 + (m0 + m + 1) * a_row_stride + b * block_size;
                    const uint8_t* q2 = act_u8 + (m0 + m + 2) * a_row_stride + b * block_size;
                    const uint8_t* q3 = act_u8 + (m0 + m + 3) * a_row_stride + b * block_size;
                    for (size_t k = 0; k < block_size; k += 4, p_k += 128) {
                        const __m512i w_lo = _mm512_loadu_si512(p_k + 0);
                        const __m512i w_hi = _mm512_loadu_si512(p_k + 64);
                        int32_t v0, v1, v2, v3;
                        std::memcpy(&v0, q0 + k, 4);
                        std::memcpy(&v1, q1 + k, 4);
                        std::memcpy(&v2, q2 + k, 4);
                        std::memcpy(&v3, q3 + k, 4);
                        const __m512i a0 = _mm512_set1_epi32(v0), a1 = _mm512_set1_epi32(v1);
                        const __m512i a2 = _mm512_set1_epi32(v2), a3 = _mm512_set1_epi32(v3);
                        d0_lo = _mm512_dpbusd_epi32(d0_lo, a0, w_lo);
                        d0_hi = _mm512_dpbusd_epi32(d0_hi, a0, w_hi);
                        d1_lo = _mm512_dpbusd_epi32(d1_lo, a1, w_lo);
                        d1_hi = _mm512_dpbusd_epi32(d1_hi, a1, w_hi);
                        d2_lo = _mm512_dpbusd_epi32(d2_lo, a2, w_lo);
                        d2_hi = _mm512_dpbusd_epi32(d2_hi, a2, w_hi);
                        d3_lo = _mm512_dpbusd_epi32(d3_lo, a3, w_lo);
                        d3_hi = _mm512_dpbusd_epi32(d3_hi, a3, w_hi);
                    }
                    const __m512i d_los[4]{d0_lo, d1_lo, d2_lo, d3_lo};
                    const __m512i d_his[4]{d0_hi, d1_hi, d2_hi, d3_hi};
                    for (size_t r = 0; r < 4; ++r) {
                        const size_t idx = (m0 + m + r) * B + b;
                        const int32_t zp_u8 = act_zp != nullptr ? static_cast<int32_t>(act_zp[idx]) : 128;
                        const __m512i vzp = _mm512_set1_epi32(zp_u8);
                        const __m512 vas = _mm512_set1_ps(act_scale[idx]);
                        _mm512_store_ps(acc_buf[m + r] + 0,
                                        _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(
                                                            d_los[r], _mm512_mullo_epi32(wsum_lo, vzp))),
                                                        _mm512_mul_ps(ws_lo, vas), _mm512_load_ps(acc_buf[m + r] + 0)));
                        _mm512_store_ps(
                            acc_buf[m + r] + 16,
                            _mm512_fmadd_ps(
                                _mm512_cvtepi32_ps(_mm512_sub_epi32(d_his[r], _mm512_mullo_epi32(wsum_hi, vzp))),
                                _mm512_mul_ps(ws_hi, vas), _mm512_load_ps(acc_buf[m + r] + 16)));
                    }
                }
                for (; m < m_count; ++m) {
                    __m512i d_lo = _mm512_setzero_si512(), d_hi = _mm512_setzero_si512();
                    const uint8_t* p_k = p_block;
                    const uint8_t* q0 = act_u8 + (m0 + m) * a_row_stride + b * block_size;
                    for (size_t k = 0; k < block_size; k += 4, p_k += 128) {
                        int32_t v0 = 0;
                        std::memcpy(&v0, q0 + k, 4);
                        const __m512i a0 = _mm512_set1_epi32(v0);
                        d_lo = _mm512_dpbusd_epi32(d_lo, a0, _mm512_loadu_si512(p_k + 0));
                        d_hi = _mm512_dpbusd_epi32(d_hi, a0, _mm512_loadu_si512(p_k + 64));
                    }
                    const size_t idx = (m0 + m) * B + b;
                    const int32_t zp_u8 = act_zp != nullptr ? static_cast<int32_t>(act_zp[idx]) : 128;
                    const __m512i vzp = _mm512_set1_epi32(zp_u8);
                    const __m512 vas = _mm512_set1_ps(act_scale[idx]);
                    _mm512_store_ps(
                        acc_buf[m] + 0,
                        _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(d_lo, _mm512_mullo_epi32(wsum_lo, vzp))),
                                        _mm512_mul_ps(ws_lo, vas), _mm512_load_ps(acc_buf[m] + 0)));
                    _mm512_store_ps(
                        acc_buf[m] + 16,
                        _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(d_hi, _mm512_mullo_epi32(wsum_hi, vzp))),
                                        _mm512_mul_ps(ws_hi, vas), _mm512_load_ps(acc_buf[m] + 16)));
                }
            }
            for (size_t m = 0; m < m_count; ++m) {
                store_row32_x86(y + (m0 + m) * y_stride + col_off, _mm512_load_ps(acc_buf[m] + 0),
                                _mm512_load_ps(acc_buf[m] + 16), n_valid, output_scale);
            }
        }
    }
}

template <bool HAS_ZERO>
YNN_X86_TARGET_VNNI void dispatch_i4_tiles_x86(size_t M, size_t N, size_t B, size_t block_size, float output_scale,
                                               const uint8_t* packed_ptr, const uint8_t* act_u8, const float* act_scale,
                                               const float* act_bsum, float* y, size_t t_begin, size_t t_end) noexcept {
    const size_t chunks = block_size / 32;
    const size_t weight_bytes = chunks * 512;
    const size_t block_bytes = weight_bytes + 64 + (HAS_ZERO ? 64 : 0) + 128;
    const size_t tile_bytes = B * block_bytes;
    const __m512i low_mask = _mm512_set1_epi8(0x0F);
    const __m512i sign_lut = _mm512_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5,
                                             -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, 7, 6,
                                             5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0);
    const __m512i v128 = _mm512_set1_epi32(128);

    if (M == 1) {
        size_t tile = t_begin;
        for (; tile + 2 <= t_end && (tile + 2) * 32 <= N; tile += 2) {
            __m512 a0_lo = _mm512_setzero_ps(), a0_hi = _mm512_setzero_ps();
            __m512 a1_lo = _mm512_setzero_ps(), a1_hi = _mm512_setzero_ps();
            const uint8_t* p0 = packed_ptr + tile * tile_bytes;
            const uint8_t* p1 = packed_ptr + (tile + 1) * tile_bytes;
            const uint8_t* qp = act_u8;
            for (size_t b = 0; b < B; ++b) {
                __m512i d0_lo = _mm512_setzero_si512(), d0_hi = _mm512_setzero_si512();
                __m512i d1_lo = _mm512_setzero_si512(), d1_hi = _mm512_setzero_si512();
                for (size_t c = 0; c < chunks; ++c, qp += 32) {
                    for (size_t step = 0; step < 4; ++step, p0 += 128, p1 += 128) {
                        int32_t qe_dw = 0, qo_dw = 0;
                        std::memcpy(&qe_dw, qp + step * 4, 4);
                        std::memcpy(&qo_dw, qp + 16 + step * 4, 4);
                        const __m512i qe = _mm512_set1_epi32(qe_dw);
                        const __m512i qo = _mm512_set1_epi32(qo_dw);
                        const __m512i r0_lo = _mm512_loadu_si512(p0 + 0), r0_hi = _mm512_loadu_si512(p0 + 64);
                        const __m512i r1_lo = _mm512_loadu_si512(p1 + 0), r1_hi = _mm512_loadu_si512(p1 + 64);
                        d0_lo = _mm512_dpbusd_epi32(d0_lo, qe,
                                                    _mm512_shuffle_epi8(sign_lut, _mm512_and_si512(r0_lo, low_mask)));
                        d0_hi = _mm512_dpbusd_epi32(d0_hi, qe,
                                                    _mm512_shuffle_epi8(sign_lut, _mm512_and_si512(r0_hi, low_mask)));
                        d1_lo = _mm512_dpbusd_epi32(d1_lo, qe,
                                                    _mm512_shuffle_epi8(sign_lut, _mm512_and_si512(r1_lo, low_mask)));
                        d1_hi = _mm512_dpbusd_epi32(d1_hi, qe,
                                                    _mm512_shuffle_epi8(sign_lut, _mm512_and_si512(r1_hi, low_mask)));
                        d0_lo = _mm512_dpbusd_epi32(
                            d0_lo, qo,
                            _mm512_shuffle_epi8(sign_lut, _mm512_and_si512(_mm512_srli_epi16(r0_lo, 4), low_mask)));
                        d0_hi = _mm512_dpbusd_epi32(
                            d0_hi, qo,
                            _mm512_shuffle_epi8(sign_lut, _mm512_and_si512(_mm512_srli_epi16(r0_hi, 4), low_mask)));
                        d1_lo = _mm512_dpbusd_epi32(
                            d1_lo, qo,
                            _mm512_shuffle_epi8(sign_lut, _mm512_and_si512(_mm512_srli_epi16(r1_lo, 4), low_mask)));
                        d1_hi = _mm512_dpbusd_epi32(
                            d1_hi, qo,
                            _mm512_shuffle_epi8(sign_lut, _mm512_and_si512(_mm512_srli_epi16(r1_hi, 4), low_mask)));
                    }
                }
                const __m512 ws0_lo = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p0 + 0)));
                const __m512 ws0_hi = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p0 + 32)));
                const __m512 ws1_lo = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p1 + 0)));
                const __m512 ws1_hi = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p1 + 32)));
                p0 += 64;
                p1 += 64;
                __m512 wz0_lo = _mm512_setzero_ps(), wz0_hi = _mm512_setzero_ps();
                __m512 wz1_lo = _mm512_setzero_ps(), wz1_hi = _mm512_setzero_ps();
                if constexpr (HAS_ZERO) {
                    wz0_lo = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p0 + 0)));
                    wz0_hi = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p0 + 32)));
                    wz1_lo = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p1 + 0)));
                    wz1_hi = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p1 + 32)));
                    p0 += 64;
                    p1 += 64;
                }
                const __m512 vas = _mm512_set1_ps(act_scale[b]);
                a0_lo = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_sub_epi32(d0_lo, _mm512_mullo_epi32(_mm512_loadu_si512(p0 + 0), v128))),
                    _mm512_mul_ps(ws0_lo, vas), a0_lo);
                a0_hi = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_sub_epi32(d0_hi, _mm512_mullo_epi32(_mm512_loadu_si512(p0 + 64), v128))),
                    _mm512_mul_ps(ws0_hi, vas), a0_hi);
                a1_lo = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_sub_epi32(d1_lo, _mm512_mullo_epi32(_mm512_loadu_si512(p1 + 0), v128))),
                    _mm512_mul_ps(ws1_lo, vas), a1_lo);
                a1_hi = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_sub_epi32(d1_hi, _mm512_mullo_epi32(_mm512_loadu_si512(p1 + 64), v128))),
                    _mm512_mul_ps(ws1_hi, vas), a1_hi);
                p0 += 128;
                p1 += 128;
                if constexpr (HAS_ZERO) {
                    const __m512 vaz = _mm512_set1_ps(act_bsum[b]);
                    a0_lo = _mm512_fmadd_ps(wz0_lo, vaz, a0_lo);
                    a0_hi = _mm512_fmadd_ps(wz0_hi, vaz, a0_hi);
                    a1_lo = _mm512_fmadd_ps(wz1_lo, vaz, a1_lo);
                    a1_hi = _mm512_fmadd_ps(wz1_hi, vaz, a1_hi);
                }
            }
            store_row32_x86(y + tile * 32 + 0, a0_lo, a0_hi, 32, output_scale);
            store_row32_x86(y + tile * 32 + 32, a1_lo, a1_hi, 32, output_scale);
        }
        for (; tile < t_end; ++tile) {
            const size_t n0 = tile * 32;
            const size_t n_valid = std::min<size_t>(32, N - n0);
            const uint8_t* tile_ptr = packed_ptr + tile * tile_bytes;
            __m512 acc_lo = _mm512_setzero_ps(), acc_hi = _mm512_setzero_ps();
            const uint8_t* p = tile_ptr;
            const uint8_t* qp = act_u8;
            for (size_t b = 0; b < B; ++b) {
                __m512i d0_lo = _mm512_setzero_si512(), d0_hi = _mm512_setzero_si512();
                __m512i d1_lo = _mm512_setzero_si512(), d1_hi = _mm512_setzero_si512();
                for (size_t c = 0; c < chunks; ++c, qp += 32) {
                    for (size_t step = 0; step < 4; ++step, p += 128) {
                        int32_t qe_dw = 0, qo_dw = 0;
                        std::memcpy(&qe_dw, qp + step * 4, 4);
                        std::memcpy(&qo_dw, qp + 16 + step * 4, 4);
                        const __m512i qe = _mm512_set1_epi32(qe_dw);
                        const __m512i qo = _mm512_set1_epi32(qo_dw);
                        const __m512i r_lo = _mm512_loadu_si512(p + 0);
                        const __m512i r_hi = _mm512_loadu_si512(p + 64);
                        const __m512i we_lo = _mm512_shuffle_epi8(sign_lut, _mm512_and_si512(r_lo, low_mask));
                        const __m512i wo_lo =
                            _mm512_shuffle_epi8(sign_lut, _mm512_and_si512(_mm512_srli_epi16(r_lo, 4), low_mask));
                        const __m512i we_hi = _mm512_shuffle_epi8(sign_lut, _mm512_and_si512(r_hi, low_mask));
                        const __m512i wo_hi =
                            _mm512_shuffle_epi8(sign_lut, _mm512_and_si512(_mm512_srli_epi16(r_hi, 4), low_mask));
                        d0_lo = _mm512_dpbusd_epi32(d0_lo, qe, we_lo);
                        d0_hi = _mm512_dpbusd_epi32(d0_hi, qe, we_hi);
                        d1_lo = _mm512_dpbusd_epi32(d1_lo, qo, wo_lo);
                        d1_hi = _mm512_dpbusd_epi32(d1_hi, qo, wo_hi);
                    }
                }
                const __m512 ws_lo = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 0)));
                const __m512 ws_hi = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32)));
                p += 64;
                __m512 wz_lo = _mm512_setzero_ps(), wz_hi = _mm512_setzero_ps();
                if constexpr (HAS_ZERO) {
                    wz_lo = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 0)));
                    wz_hi = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32)));
                    p += 64;
                }
                const __m512i wsum_lo = _mm512_loadu_si512(p + 0);
                const __m512i wsum_hi = _mm512_loadu_si512(p + 64);
                p += 128;
                const __m512 vas = _mm512_set1_ps(act_scale[b]);
                const __m512i corr_lo =
                    _mm512_sub_epi32(_mm512_add_epi32(d0_lo, d1_lo), _mm512_mullo_epi32(wsum_lo, v128));
                const __m512i corr_hi =
                    _mm512_sub_epi32(_mm512_add_epi32(d0_hi, d1_hi), _mm512_mullo_epi32(wsum_hi, v128));
                acc_lo = _mm512_fmadd_ps(_mm512_cvtepi32_ps(corr_lo), _mm512_mul_ps(ws_lo, vas), acc_lo);
                acc_hi = _mm512_fmadd_ps(_mm512_cvtepi32_ps(corr_hi), _mm512_mul_ps(ws_hi, vas), acc_hi);
                if constexpr (HAS_ZERO) {
                    const __m512 vaz = _mm512_set1_ps(act_bsum[b]);
                    acc_lo = _mm512_fmadd_ps(wz_lo, vaz, acc_lo);
                    acc_hi = _mm512_fmadd_ps(wz_hi, vaz, acc_hi);
                }
            }
            store_row32_x86(y + n0, acc_lo, acc_hi, n_valid, output_scale);
        }
        return;
    }

    for (size_t tile = t_begin; tile < t_end; ++tile) {
        const size_t n0 = tile * 32;
        const size_t n_valid = std::min<size_t>(32, N - n0);
        const uint8_t* tile_ptr = packed_ptr + tile * tile_bytes;
        constexpr size_t M_TILE = 32;
        alignas(64) float acc_buf[M_TILE][32];
        for (size_t m0 = 0; m0 < M; m0 += M_TILE) {
            const size_t m_count = std::min<size_t>(M_TILE, M - m0);
            std::memset(acc_buf, 0, m_count * 32 * sizeof(float));
            const uint8_t* p_block = tile_ptr;
            for (size_t b = 0; b < B; ++b, p_block += block_bytes) {
                const uint8_t* meta = p_block + weight_bytes;
                const __m512 ws_lo = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(meta + 0)));
                const __m512 ws_hi = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(meta + 32)));
                meta += 64;
                __m512 wz_lo = _mm512_setzero_ps(), wz_hi = _mm512_setzero_ps();
                if constexpr (HAS_ZERO) {
                    wz_lo = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(meta + 0)));
                    wz_hi = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(meta + 32)));
                    meta += 64;
                }
                const __m512i wsum_lo = _mm512_loadu_si512(meta + 0);
                const __m512i wsum_hi = _mm512_loadu_si512(meta + 64);

                for (size_t m = 0; m < m_count; ++m) {
                    __m512i d0_lo = _mm512_setzero_si512(), d0_hi = _mm512_setzero_si512();
                    __m512i d1_lo = _mm512_setzero_si512(), d1_hi = _mm512_setzero_si512();
                    const uint8_t* p = p_block;
                    const uint8_t* qp = act_u8 + ((m0 + m) * B + b) * block_size;
                    for (size_t c = 0; c < chunks; ++c, qp += 32) {
                        for (size_t step = 0; step < 4; ++step, p += 128) {
                            int32_t qe_dw = 0, qo_dw = 0;
                            std::memcpy(&qe_dw, qp + step * 4, 4);
                            std::memcpy(&qo_dw, qp + 16 + step * 4, 4);
                            const __m512i qe = _mm512_set1_epi32(qe_dw);
                            const __m512i qo = _mm512_set1_epi32(qo_dw);
                            const __m512i r_lo = _mm512_loadu_si512(p + 0);
                            const __m512i r_hi = _mm512_loadu_si512(p + 64);
                            const __m512i we_lo = _mm512_shuffle_epi8(sign_lut, _mm512_and_si512(r_lo, low_mask));
                            const __m512i wo_lo =
                                _mm512_shuffle_epi8(sign_lut, _mm512_and_si512(_mm512_srli_epi16(r_lo, 4), low_mask));
                            const __m512i we_hi = _mm512_shuffle_epi8(sign_lut, _mm512_and_si512(r_hi, low_mask));
                            const __m512i wo_hi =
                                _mm512_shuffle_epi8(sign_lut, _mm512_and_si512(_mm512_srli_epi16(r_hi, 4), low_mask));
                            d0_lo = _mm512_dpbusd_epi32(d0_lo, qe, we_lo);
                            d0_hi = _mm512_dpbusd_epi32(d0_hi, qe, we_hi);
                            d1_lo = _mm512_dpbusd_epi32(d1_lo, qo, wo_lo);
                            d1_hi = _mm512_dpbusd_epi32(d1_hi, qo, wo_hi);
                        }
                    }
                    const size_t idx = (m0 + m) * B + b;
                    const __m512 vas = _mm512_set1_ps(act_scale[idx]);
                    const __m512i corr_lo =
                        _mm512_sub_epi32(_mm512_add_epi32(d0_lo, d1_lo), _mm512_mullo_epi32(wsum_lo, v128));
                    const __m512i corr_hi =
                        _mm512_sub_epi32(_mm512_add_epi32(d0_hi, d1_hi), _mm512_mullo_epi32(wsum_hi, v128));
                    __m512 a_lo = _mm512_fmadd_ps(_mm512_cvtepi32_ps(corr_lo), _mm512_mul_ps(ws_lo, vas),
                                                  _mm512_load_ps(acc_buf[m] + 0));
                    __m512 a_hi = _mm512_fmadd_ps(_mm512_cvtepi32_ps(corr_hi), _mm512_mul_ps(ws_hi, vas),
                                                  _mm512_load_ps(acc_buf[m] + 16));
                    if constexpr (HAS_ZERO) {
                        const __m512 vaz = _mm512_set1_ps(act_bsum[idx]);
                        a_lo = _mm512_fmadd_ps(wz_lo, vaz, a_lo);
                        a_hi = _mm512_fmadd_ps(wz_hi, vaz, a_hi);
                    }
                    _mm512_store_ps(acc_buf[m] + 0, a_lo);
                    _mm512_store_ps(acc_buf[m] + 16, a_hi);
                }
            }
            for (size_t m = 0; m < m_count; ++m) {
                store_row32_x86(y + (m0 + m) * N + n0, _mm512_load_ps(acc_buf[m] + 0), _mm512_load_ps(acc_buf[m] + 16),
                                n_valid, output_scale);
            }
        }
    }
}

YNN_X86_TARGET_VNNI void quantize_blockwise_blocks_x86(int32_t bits, size_t k, size_t block_size, bool compute_bsum,
                                                       const float* input, uint8_t* quantized_input,
                                                       float* input_scales, float* input_bsums, size_t block_start,
                                                       size_t block_end) noexcept {
    (void)k;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i v128 = _mm512_set1_epi32(128);

    if (bits == 4) {
        const size_t chunks = block_size / 32;
        const __m512i even_idx = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        const __m512i odd_idx = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
        for (size_t fb = block_start; fb < block_end; ++fb) {
            const float* x_block = input + fb * block_size;
            __m512 vmax0 = _mm512_setzero_ps(), vmax1 = _mm512_setzero_ps();
            __m512 vsum0 = _mm512_setzero_ps(), vsum1 = _mm512_setzero_ps();
            for (size_t c = 0; c < chunks; ++c) {
                const float* xp = x_block + c * 32;
                const __m512 x0 = _mm512_loadu_ps(xp + 0);
                const __m512 x1 = _mm512_loadu_ps(xp + 16);
                if (compute_bsum) {
                    vsum0 = _mm512_add_ps(vsum0, x0);
                    vsum1 = _mm512_add_ps(vsum1, x1);
                }
                vmax0 = _mm512_max_ps(vmax0, _mm512_abs_ps(x0));
                vmax1 = _mm512_max_ps(vmax1, _mm512_abs_ps(x1));
            }
            const float amax = _mm512_reduce_max_ps(_mm512_max_ps(vmax0, vmax1));
            const float scale = amax / 127.0f;
            const float inv_scale = amax > 0.0f ? 127.0f / amax : 0.0f;
            input_scales[fb] = scale;
            if (compute_bsum) input_bsums[fb] = _mm512_reduce_add_ps(_mm512_add_ps(vsum0, vsum1));

            const __m512 vinv = _mm512_set1_ps(inv_scale);
            uint8_t* qp_base = quantized_input + fb * block_size;
            for (size_t c = 0; c < chunks; ++c) {
                const float* xp = x_block + c * 32;
                const __m512i q0 =
                    _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(xp + 0), vinv)), v128);
                const __m512i q1 =
                    _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(xp + 16), vinv)), v128);
                const __m512i qe = _mm512_max_epi32(_mm512_permutex2var_epi32(q0, even_idx, q1), zero);
                const __m512i qo = _mm512_max_epi32(_mm512_permutex2var_epi32(q0, odd_idx, q1), zero);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(qp_base + c * 32 + 0), _mm512_cvtusepi32_epi8(qe));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(qp_base + c * 32 + 16), _mm512_cvtusepi32_epi8(qo));
            }
        }
        return;
    }

    if (bits == 8) {
        for (size_t fb = block_start; fb < block_end; ++fb) {
            const float* x_block = input + fb * block_size;
            if (compute_bsum && input_bsums != nullptr) {
                __m512 vmin0 = _mm512_setzero_ps(), vmin1 = _mm512_setzero_ps();
                __m512 vmax0 = _mm512_setzero_ps(), vmax1 = _mm512_setzero_ps();
                size_t c = 0;
                for (; c + 32 <= block_size; c += 32) {
                    const __m512 x0 = _mm512_loadu_ps(x_block + c + 0);
                    const __m512 x1 = _mm512_loadu_ps(x_block + c + 16);
                    vmin0 = _mm512_min_ps(vmin0, x0);
                    vmin1 = _mm512_min_ps(vmin1, x1);
                    vmax0 = _mm512_max_ps(vmax0, x0);
                    vmax1 = _mm512_max_ps(vmax1, x1);
                }
                for (; c < block_size; c += 16) {
                    const __m512 x0 = _mm512_loadu_ps(x_block + c);
                    vmin0 = _mm512_min_ps(vmin0, x0);
                    vmax0 = _mm512_max_ps(vmax0, x0);
                }
                const float rmin = _mm512_reduce_min_ps(_mm512_min_ps(vmin0, vmin1));
                const float rmax = _mm512_reduce_max_ps(_mm512_max_ps(vmax0, vmax1));
                const float rdiff = rmax - rmin;
                float scale = 1.0f;
                float inv_scale = 0.0f;
                int32_t zp_s8 = -128;
                if (rdiff > 0.0f) {
                    inv_scale = 255.0f / rdiff;
                    scale = rdiff / 255.0f;
                    const float min_error = -128.0f + rmin * inv_scale;
                    const float max_error = 127.0f + rmax * inv_scale;
                    const float fzp =
                        min_error + max_error > 0.0f ? -128.0f - rmin * inv_scale : 127.0f - rmax * inv_scale;
                    zp_s8 = static_cast<int32_t>(std::rint(std::clamp(fzp, -128.0f, 127.0f)));
                }
                const int32_t zp_u8 = zp_s8 + 128;
                input_scales[fb] = scale;
                input_bsums[fb] = static_cast<float>(zp_u8);
                const __m512 vinv = _mm512_set1_ps(inv_scale);
                const __m512 vhalf = _mm512_set1_ps(0.5f);
                const __m512i vone = _mm512_set1_epi32(1);
                const __m512i vzp = _mm512_set1_epi32(zp_u8);
                uint8_t* qp = quantized_input + fb * block_size;
                for (size_t col = 0; col < block_size; col += 16) {
                    const __m512 prod = _mm512_mul_ps(_mm512_loadu_ps(x_block + col), vinv);
                    const __mmask16 neg = _mm512_movepi32_mask(_mm512_castps_si512(prod));
                    const __m512 vabs = _mm512_abs_ps(prod);
                    const __m512 trunc_abs = _mm512_roundscale_ps(vabs, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
                    const __mmask16 round_up = _mm512_cmp_ps_mask(_mm512_sub_ps(vabs, trunc_abs), vhalf, _CMP_GE_OQ);
                    __m512i qi = _mm512_cvttps_epi32(trunc_abs);
                    qi = _mm512_mask_add_epi32(qi, round_up, qi, vone);
                    qi = _mm512_mask_sub_epi32(qi, neg, zero, qi);
                    const __m512i q32 = _mm512_max_epi32(_mm512_add_epi32(qi, vzp), zero);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(qp + col), _mm512_cvtusepi32_epi8(q32));
                }
                continue;
            }

            __m512 vmax0 = _mm512_setzero_ps(), vmax1 = _mm512_setzero_ps();
            size_t c = 0;
            for (; c + 32 <= block_size; c += 32) {
                vmax0 = _mm512_max_ps(vmax0, _mm512_abs_ps(_mm512_loadu_ps(x_block + c + 0)));
                vmax1 = _mm512_max_ps(vmax1, _mm512_abs_ps(_mm512_loadu_ps(x_block + c + 16)));
            }
            for (; c < block_size; c += 16) {
                vmax0 = _mm512_max_ps(vmax0, _mm512_abs_ps(_mm512_loadu_ps(x_block + c)));
            }
            const float amax = _mm512_reduce_max_ps(_mm512_max_ps(vmax0, vmax1));
            const float scale = amax / 127.0f;
            const float inv_scale = amax > 0.0f ? 127.0f / amax : 0.0f;
            input_scales[fb] = scale;
            const __m512 vinv = _mm512_set1_ps(inv_scale);
            const __m512i vmin_u8 = _mm512_set1_epi32(1);
            uint8_t* qp = quantized_input + fb * block_size;
            for (size_t col = 0; col < block_size; col += 16) {
                const __m512 xv = _mm512_loadu_ps(x_block + col);
                const __m512i q32 = _mm512_max_epi32(
                    _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(xv, vinv)), v128), amax > 0.0f ? vmin_u8 : v128);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(qp + col), _mm512_cvtusepi32_epi8(q32));
            }
        }
    }
}

YNN_X86_TARGET_VNNI void quantize_symmetric_blocks_x86(int32_t bits, size_t k, size_t block_size, float static_scale,
                                                       const float* input, uint8_t* quantized_input,
                                                       float* input_scales, size_t block_start, size_t block_end,
                                                       size_t B, size_t padded_block) noexcept {
    const size_t chunks = padded_block / 32;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i v128 = _mm512_set1_epi32(128);
    const __m512i even_idx = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i odd_idx = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    const __m128i i2_shuf = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

    for (size_t fb = block_start; fb < block_end; ++fb) {
        const size_t row = fb / B;
        const size_t b = fb % B;
        const float* x_block = input + row * k + b * block_size;
        float scale = static_scale;
        float inv_scale = 0.0f;
        if (static_scale <= 0.0f) {
            __m512 vmax0 = _mm512_setzero_ps(), vmax1 = _mm512_setzero_ps();
            size_t col = 0;
            for (; col + 32 <= block_size; col += 32) {
                vmax0 = _mm512_max_ps(vmax0, _mm512_abs_ps(_mm512_loadu_ps(x_block + col + 0)));
                vmax1 = _mm512_max_ps(vmax1, _mm512_abs_ps(_mm512_loadu_ps(x_block + col + 16)));
            }
            float amax = _mm512_reduce_max_ps(_mm512_max_ps(vmax0, vmax1));
            for (; col < block_size; ++col) amax = std::max(amax, std::abs(x_block[col]));
            scale = amax / 127.0f;
            inv_scale = amax > 0.0f ? 127.0f / amax : 0.0f;
        }
        input_scales[fb] = scale;
        const __m512 vscale = _mm512_set1_ps(scale);
        const __m512 vinv = _mm512_set1_ps(inv_scale);
        uint8_t* qp_base = quantized_input + fb * padded_block;

        for (size_t c = 0; c < chunks; ++c) {
            const size_t col0 = c * 32;
            __m512i q0 = v128, q1 = v128;
            if (col0 + 32 <= block_size) {
                const __m512 x0 = _mm512_loadu_ps(x_block + col0 + 0);
                const __m512 x1 = _mm512_loadu_ps(x_block + col0 + 16);
                const __m512 s0 = static_scale > 0.0f ? _mm512_div_ps(x0, vscale) : _mm512_mul_ps(x0, vinv);
                const __m512 s1 = static_scale > 0.0f ? _mm512_div_ps(x1, vscale) : _mm512_mul_ps(x1, vinv);
                q0 = _mm512_max_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(s0), v128), zero);
                q1 = _mm512_max_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(s1), v128), zero);
            } else {
                alignas(64) int32_t tmp[32];
                std::fill_n(tmp, 32, 128);
                for (size_t i = 0; i < 32 && col0 + i < block_size; ++i) {
                    const float v = static_scale > 0.0f ? (x_block[col0 + i] / scale) : (x_block[col0 + i] * inv_scale);
                    tmp[i] = static_cast<int32_t>(std::clamp(std::nearbyint(v), -128.0f, 127.0f)) + 128;
                }
                q0 = _mm512_load_si512(tmp + 0);
                q1 = _mm512_load_si512(tmp + 16);
            }
            uint8_t* qp = qp_base + c * 32;
            if (bits == 8) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(qp + 0), _mm512_cvtusepi32_epi8(q0));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(qp + 16), _mm512_cvtusepi32_epi8(q1));
            } else if (bits == 4) {
                const __m512i qe = _mm512_permutex2var_epi32(q0, even_idx, q1);
                const __m512i qo = _mm512_permutex2var_epi32(q0, odd_idx, q1);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(qp + 0), _mm512_cvtusepi32_epi8(qe));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(qp + 16), _mm512_cvtusepi32_epi8(qo));
            } else {
                const __m128i b0 = _mm_shuffle_epi8(_mm512_cvtusepi32_epi8(q0), i2_shuf);
                const __m128i b1 = _mm_shuffle_epi8(_mm512_cvtusepi32_epi8(q1), i2_shuf);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(qp + 0), b0);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(qp + 16), b1);
            }
        }
    }
}

YNN_X86_TARGET_VNNI void dispatch_symmetric_lowbit_x86(int32_t bits, size_t M, size_t N, size_t B, size_t padded_block,
                                                       float output_scale, const uint8_t* packed_ptr,
                                                       const uint8_t* act_u8, const float* act_scale, float* y,
                                                       size_t y_stride, size_t t_begin, size_t t_end) noexcept {
    const size_t chunks = padded_block / 32;
    const size_t weight_bytes = chunks * static_cast<size_t>(bits) * 128;
    const size_t block_bytes = weight_bytes + 256;
    const size_t tile_bytes = B * block_bytes;
    const __m512i v128 = _mm512_set1_epi32(128);
    const __m512i mask4 = _mm512_set1_epi8(0x0F);
    const __m512i lut4 = _mm512_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6,
                                         -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3,
                                         2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0);
    const __m512i mask2 = _mm512_set1_epi8(0x03);
    const __m512i lut2 = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, -1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, 1, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, 1, 0);

    constexpr size_t M_TILE = 32;
    alignas(64) float acc_buf[M_TILE][32];
    for (size_t tile = t_begin; tile < t_end; ++tile) {
        const size_t n0 = tile * 32;
        const size_t n_valid = std::min<size_t>(32, N - n0);
        const size_t col_off = (y_stride == 32 && t_end == t_begin + 1) ? 0 : n0;
        const uint8_t* tile_ptr = packed_ptr + tile * tile_bytes;

        for (size_t m0 = 0; m0 < M; m0 += M_TILE) {
            const size_t m_count = std::min<size_t>(M_TILE, M - m0);
            std::memset(acc_buf, 0, m_count * 32 * sizeof(float));
            const uint8_t* p_block = tile_ptr;
            for (size_t b = 0; b < B; ++b, p_block += block_bytes) {
                const auto* sp = reinterpret_cast<const float*>(p_block + weight_bytes);
                const __m512 ws_lo = _mm512_loadu_ps(sp + 0);
                const __m512 ws_hi = _mm512_loadu_ps(sp + 16);
                const __m512i wsum_lo = _mm512_loadu_si512(p_block + weight_bytes + 128);
                const __m512i wsum_hi = _mm512_loadu_si512(p_block + weight_bytes + 192);

                for (size_t m = 0; m < m_count; ++m) {
                    __m512i d0_lo = _mm512_setzero_si512(), d0_hi = _mm512_setzero_si512();
                    __m512i d1_lo = _mm512_setzero_si512(), d1_hi = _mm512_setzero_si512();
                    const uint8_t* p = p_block;
                    const uint8_t* qp = act_u8 + ((m0 + m) * B + b) * padded_block;
                    if (bits == 4) {
                        for (size_t c = 0; c < chunks; ++c, qp += 32) {
                            for (size_t step = 0; step < 4; ++step, p += 128) {
                                int32_t qe_dw = 0, qo_dw = 0;
                                std::memcpy(&qe_dw, qp + step * 4, 4);
                                std::memcpy(&qo_dw, qp + 16 + step * 4, 4);
                                const __m512i qe = _mm512_set1_epi32(qe_dw);
                                const __m512i qo = _mm512_set1_epi32(qo_dw);
                                const __m512i r_lo = _mm512_loadu_si512(p + 0);
                                const __m512i r_hi = _mm512_loadu_si512(p + 64);
                                d0_lo = _mm512_dpbusd_epi32(d0_lo, qe,
                                                            _mm512_shuffle_epi8(lut4, _mm512_and_si512(r_lo, mask4)));
                                d0_hi = _mm512_dpbusd_epi32(d0_hi, qe,
                                                            _mm512_shuffle_epi8(lut4, _mm512_and_si512(r_hi, mask4)));
                                d1_lo = _mm512_dpbusd_epi32(
                                    d1_lo, qo,
                                    _mm512_shuffle_epi8(lut4, _mm512_and_si512(_mm512_srli_epi16(r_lo, 4), mask4)));
                                d1_hi = _mm512_dpbusd_epi32(
                                    d1_hi, qo,
                                    _mm512_shuffle_epi8(lut4, _mm512_and_si512(_mm512_srli_epi16(r_hi, 4), mask4)));
                            }
                        }
                    } else {
                        for (size_t c = 0; c < chunks; ++c, qp += 32) {
                            for (size_t s = 0; s < 2; ++s, p += 128) {
                                int32_t q0_dw = 0, q1_dw = 0, q2_dw = 0, q3_dw = 0;
                                std::memcpy(&q0_dw, qp + s * 16 + 0, 4);
                                std::memcpy(&q1_dw, qp + s * 16 + 4, 4);
                                std::memcpy(&q2_dw, qp + s * 16 + 8, 4);
                                std::memcpy(&q3_dw, qp + s * 16 + 12, 4);
                                const __m512i qa0 = _mm512_set1_epi32(q0_dw), qa1 = _mm512_set1_epi32(q1_dw);
                                const __m512i qa2 = _mm512_set1_epi32(q2_dw), qa3 = _mm512_set1_epi32(q3_dw);
                                const __m512i r_lo = _mm512_loadu_si512(p + 0);
                                const __m512i r_hi = _mm512_loadu_si512(p + 64);
                                d0_lo = _mm512_dpbusd_epi32(d0_lo, qa0,
                                                            _mm512_shuffle_epi8(lut2, _mm512_and_si512(r_lo, mask2)));
                                d0_hi = _mm512_dpbusd_epi32(d0_hi, qa0,
                                                            _mm512_shuffle_epi8(lut2, _mm512_and_si512(r_hi, mask2)));
                                d1_lo = _mm512_dpbusd_epi32(
                                    d1_lo, qa1,
                                    _mm512_shuffle_epi8(lut2, _mm512_and_si512(_mm512_srli_epi16(r_lo, 2), mask2)));
                                d1_hi = _mm512_dpbusd_epi32(
                                    d1_hi, qa1,
                                    _mm512_shuffle_epi8(lut2, _mm512_and_si512(_mm512_srli_epi16(r_hi, 2), mask2)));
                                d0_lo = _mm512_dpbusd_epi32(
                                    d0_lo, qa2,
                                    _mm512_shuffle_epi8(lut2, _mm512_and_si512(_mm512_srli_epi16(r_lo, 4), mask2)));
                                d0_hi = _mm512_dpbusd_epi32(
                                    d0_hi, qa2,
                                    _mm512_shuffle_epi8(lut2, _mm512_and_si512(_mm512_srli_epi16(r_hi, 4), mask2)));
                                d1_lo = _mm512_dpbusd_epi32(
                                    d1_lo, qa3,
                                    _mm512_shuffle_epi8(lut2, _mm512_and_si512(_mm512_srli_epi16(r_lo, 6), mask2)));
                                d1_hi = _mm512_dpbusd_epi32(
                                    d1_hi, qa3,
                                    _mm512_shuffle_epi8(lut2, _mm512_and_si512(_mm512_srli_epi16(r_hi, 6), mask2)));
                            }
                        }
                    }
                    const __m512 vas = _mm512_set1_ps(act_scale[(m0 + m) * B + b]);
                    const __m512i corr_lo =
                        _mm512_sub_epi32(_mm512_add_epi32(d0_lo, d1_lo), _mm512_mullo_epi32(wsum_lo, v128));
                    const __m512i corr_hi =
                        _mm512_sub_epi32(_mm512_add_epi32(d0_hi, d1_hi), _mm512_mullo_epi32(wsum_hi, v128));
                    _mm512_store_ps(acc_buf[m] + 0,
                                    _mm512_fmadd_ps(_mm512_cvtepi32_ps(corr_lo), _mm512_mul_ps(ws_lo, vas),
                                                    _mm512_load_ps(acc_buf[m] + 0)));
                    _mm512_store_ps(acc_buf[m] + 16,
                                    _mm512_fmadd_ps(_mm512_cvtepi32_ps(corr_hi), _mm512_mul_ps(ws_hi, vas),
                                                    _mm512_load_ps(acc_buf[m] + 16)));
                }
            }
            for (size_t m = 0; m < m_count; ++m) {
                store_row32_x86(y + (m0 + m) * y_stride + col_off, _mm512_load_ps(acc_buf[m] + 0),
                                _mm512_load_ps(acc_buf[m] + 16), n_valid, output_scale);
            }
        }
    }
}

} // namespace

bool arch_is_blockwise_linear_supported(int32_t bits) noexcept { return (bits == 4 || bits == 8) && has_x86_vnni(); }

size_t arch_packed_blockwise_linear_size(int32_t bits, size_t n, size_t k, size_t block_size, bool has_zero) noexcept {
    const size_t tiles = blockwise_linear_tile_count(n);
    const size_t B = k / block_size;
    if (bits == 4) {
        const size_t chunks = block_size / 32;
        const size_t block_bytes = chunks * 512 + 64 + (has_zero ? 64 : 0) + 128;
        return tiles * B * block_bytes;
    }
    if (bits == 8) {
        const size_t block_bytes = block_size * 32 + 256;
        return tiles * B * block_bytes;
    }
    return 0;
}

void arch_pack_blockwise_linear_tiles(int32_t bits, size_t n, size_t k, size_t block_size, const void* weights,
                                      const void* scales, bool scales_fp16, const void* zeros, bool zeros_fp16,
                                      size_t tile_start, size_t tile_end, void* packed_tiles) {
    const size_t B = k / block_size;
    const size_t tiles = blockwise_linear_tile_count(n);
    tile_end = std::min(tile_end, tiles);
    auto* dst_all = static_cast<uint8_t*>(packed_tiles);
    const auto* s_f32 = scales_fp16 ? nullptr : static_cast<const float*>(scales);
    const auto* s_u16 = scales_fp16 ? static_cast<const uint16_t*>(scales) : nullptr;

    if (bits == 4) {
        const size_t row_bytes = k / 2;
        const size_t chunks = block_size / 32;
        const bool has_zero = zeros != nullptr;
        const size_t block_bytes = chunks * 512 + 64 + (has_zero ? 64 : 0) + 128;
        const size_t tile_bytes = B * block_bytes;
        const auto* w_all = static_cast<const uint8_t*>(weights);
        const auto* z_f32 = (has_zero && !zeros_fp16) ? static_cast<const float*>(zeros) : nullptr;
        const auto* z_u16 = (has_zero && zeros_fp16) ? static_cast<const uint16_t*>(zeros) : nullptr;

        for (size_t tile = tile_start; tile < tile_end; ++tile) {
            const size_t n0 = tile * 32;
            uint8_t* dst = dst_all + tile * tile_bytes;
            for (size_t b = 0; b < B; ++b) {
                int32_t wsum[32] = {};
                for (size_t c = 0; c < chunks; ++c) {
                    const size_t col_byte_base = (b * chunks + c) * 16;
                    for (size_t step = 0; step < 4; ++step) {
                        const size_t off = col_byte_base + step * 4;
                        for (size_t r = 0; r < 32; ++r) {
                            const size_t row = n0 + r;
                            uint32_t word = 0;
                            if (row < n) {
                                const uint8_t* wp = w_all + row * row_bytes + off;
                                std::memcpy(&word, wp, sizeof(uint32_t));
                                for (size_t bi = 0; bi < 4; ++bi) {
                                    const auto raw = wp[bi];
                                    const int32_t lo = static_cast<int8_t>(raw << 4) >> 4;
                                    const int32_t hi = static_cast<int8_t>(raw) >> 4;
                                    wsum[r] += lo + hi;
                                }
                            }
                            std::memcpy(dst + r * 4, &word, sizeof(uint32_t));
                        }
                        dst += 128;
                    }
                }
                auto* dst_s = reinterpret_cast<uint16_t*>(dst);
                for (size_t r = 0; r < 32; ++r) {
                    const size_t row = n0 + r;
                    const size_t idx = row * B + b;
                    dst_s[r] = row < n ? (scales_fp16 ? s_u16[idx] : half(s_f32[idx]).to_bits()) : uint16_t{0};
                }
                dst += 64;
                if (has_zero) {
                    auto* dst_z = reinterpret_cast<uint16_t*>(dst);
                    for (size_t r = 0; r < 32; ++r) {
                        const size_t row = n0 + r;
                        const size_t idx = row * B + b;
                        dst_z[r] = row < n ? (zeros_fp16 ? z_u16[idx] : half(z_f32[idx]).to_bits()) : uint16_t{0};
                    }
                    dst += 64;
                }
                std::memcpy(dst, wsum, 32 * sizeof(int32_t));
                dst += 128;
            }
        }
        return;
    }

    if (bits == 8) {
        const size_t block_bytes = block_size * 32 + 256;
        const size_t tile_bytes = B * block_bytes;
        const auto* w_all = static_cast<const int8_t*>(weights);
        for (size_t tile = tile_start; tile < tile_end; ++tile) {
            const size_t n0 = tile * 32;
            uint8_t* dst = dst_all + tile * tile_bytes;
            for (size_t b = 0; b < B; ++b) {
                const size_t col_base = b * block_size;
                int32_t wsum[32] = {};
                for (size_t k_idx = 0; k_idx < block_size; k_idx += 4) {
                    for (size_t r = 0; r < 32; ++r) {
                        const size_t row = n0 + r;
                        int32_t word = 0;
                        if (row < n) {
                            const int8_t* wp = w_all + row * k + col_base + k_idx;
                            std::memcpy(&word, wp, sizeof(int32_t));
                            wsum[r] += static_cast<int32_t>(wp[0]) + static_cast<int32_t>(wp[1]) +
                                       static_cast<int32_t>(wp[2]) + static_cast<int32_t>(wp[3]);
                        }
                        std::memcpy(dst + r * 4, &word, sizeof(int32_t));
                    }
                    dst += 128;
                }
                auto* dst_s = reinterpret_cast<float*>(dst);
                for (size_t r = 0; r < 32; ++r) {
                    const size_t row = n0 + r;
                    const size_t idx = row * B + b;
                    dst_s[r] =
                        row < n ? (scales_fp16 ? static_cast<float>(half::from_bits(s_u16[idx])) : s_f32[idx]) : 0.0f;
                }
                dst += 128;
                std::memcpy(dst, wsum, 32 * sizeof(int32_t));
                dst += 128;
            }
        }
    }
}

void arch_quantize_blockwise_blocks_f32(int32_t bits, size_t m, size_t k, size_t block_size, bool compute_bsum,
                                        const float* input, int8_t* quantized_input, float* input_scales,
                                        float* input_bsums, size_t block_start, size_t block_end) {
    if (!has_x86_vnni()) return;
    const size_t B = k / block_size;
    block_end = std::min(block_end, m * B);
    if (block_start >= block_end) return;
    quantize_blockwise_blocks_x86(bits, k, block_size, compute_bsum, input, reinterpret_cast<uint8_t*>(quantized_input),
                                  input_scales, input_bsums, block_start, block_end);
}

void arch_blockwise_linear_f32(int32_t bits, size_t m, size_t n, size_t k, size_t block_size, bool has_zero,
                               float output_scale, const void* packed_tiles, const int8_t* quantized_input,
                               const float* input_scales, const float* input_bsums, float* output, size_t tile_start,
                               size_t tile_end) {
    if (!has_x86_vnni() || m == 0) return;
    const size_t tiles = blockwise_linear_tile_count(n);
    tile_end = std::min(tile_end, tiles);
    if (tile_start >= tile_end) return;
    const size_t B = k / block_size;
    const auto* ptr = static_cast<const uint8_t*>(packed_tiles);
    const auto* act_u8 = reinterpret_cast<const uint8_t*>(quantized_input);

    if (bits == 4) {
        if (has_zero) {
            dispatch_i4_tiles_x86<true>(m, n, B, block_size, output_scale, ptr, act_u8, input_scales, input_bsums,
                                        output, tile_start, tile_end);
        } else {
            dispatch_i4_tiles_x86<false>(m, n, B, block_size, output_scale, ptr, act_u8, input_scales, input_bsums,
                                         output, tile_start, tile_end);
        }
        return;
    }

    if (bits == 8) {
        if (m >= 2 && block_size >= 64 && has_x86_amx_int8()) {
            dispatch_i8_tiles_amx(m, n, B, k, block_size, output_scale, ptr, act_u8, input_scales, input_bsums, output,
                                  n, tile_start, tile_end);
        } else {
            dispatch_i8_tiles_vnni(m, n, B, k, block_size, output_scale, ptr, act_u8, input_scales, input_bsums, output,
                                   n, tile_start, tile_end);
        }
    }
}

bool arch_is_symmetric_linear_supported(int32_t bits) noexcept {
    return (bits == 2 || bits == 4 || bits == 8) && has_x86_vnni();
}

size_t arch_packed_symmetric_linear_size(int32_t bits, size_t n, size_t k, size_t block_size) noexcept {
    const size_t tiles = blockwise_linear_tile_count(n);
    const size_t B = k / block_size;
    const size_t chunks = symmetric_linear_padded_block(block_size) / 32;
    const size_t weight_bytes = chunks * static_cast<size_t>(bits) * 128;
    return tiles * B * (weight_bytes + 256);
}

void arch_pack_symmetric_linear_tiles(int32_t bits, size_t n, size_t k, size_t row_offset, size_t block_size,
                                      const void* weights, const void* scales, bool scales_fp16, size_t tile_start,
                                      size_t tile_end, void* packed_tiles) {
    const size_t tiles = blockwise_linear_tile_count(n);
    tile_end = std::min(tile_end, tiles);
    if (tile_start >= tile_end) return;

    const size_t values_per_byte = static_cast<size_t>(8 / bits);
    const size_t row_bytes = k / values_per_byte;
    const size_t block_row_bytes = block_size / values_per_byte;
    const size_t B = k / block_size;
    const size_t chunks = symmetric_linear_padded_block(block_size) / 32;
    const size_t steps_per_chunk = static_cast<size_t>(bits);
    const size_t weight_bytes = chunks * steps_per_chunk * 128;
    const size_t block_bytes = weight_bytes + 256;
    const size_t tile_bytes = B * block_bytes;
    const uint8_t mask = static_cast<uint8_t>((1 << bits) - 1);
    const int32_t sign_bit = 1 << (bits - 1);
    const int32_t full_range = 1 << bits;

    auto* dst_all = static_cast<uint8_t*>(packed_tiles);
    const auto* w_all = static_cast<const uint8_t*>(weights);
    const auto* s_f32 = scales_fp16 ? nullptr : static_cast<const float*>(scales);
    const auto* s_u16 = scales_fp16 ? static_cast<const uint16_t*>(scales) : nullptr;

    for (size_t tile = tile_start; tile < tile_end; ++tile) {
        const size_t n0 = tile * 32;
        uint8_t* dst = dst_all + tile * tile_bytes;
        for (size_t b = 0; b < B; ++b) {
            int32_t wsum[32] = {};
            for (size_t c = 0; c < chunks; ++c) {
                for (size_t step = 0; step < steps_per_chunk; ++step) {
                    const size_t byte_in_block = (c * steps_per_chunk + step) * 4;
                    const size_t valid_bytes =
                        byte_in_block < block_row_bytes ? std::min<size_t>(4, block_row_bytes - byte_in_block) : 0;
                    for (size_t r = 0; r < 32; ++r) {
                        const size_t row = n0 + r;
                        uint32_t word = 0;
                        if (row < n && valid_bytes > 0) {
                            const uint8_t* wp =
                                w_all + (row_offset + row) * row_bytes + b * block_row_bytes + byte_in_block;
                            std::memcpy(&word, wp, valid_bytes);
                            for (size_t bi = 0; bi < valid_bytes; ++bi) {
                                const uint8_t byte = wp[bi];
                                for (size_t v = 0; v < values_per_byte; ++v) {
                                    int32_t code = static_cast<int32_t>((byte >> (v * bits)) & mask);
                                    if (code & sign_bit) code -= full_range;
                                    wsum[r] += code;
                                }
                            }
                        }
                        std::memcpy(dst + r * 4, &word, sizeof(uint32_t));
                    }
                    dst += 128;
                }
            }
            auto* dst_s = reinterpret_cast<float*>(dst);
            for (size_t r = 0; r < 32; ++r) {
                const size_t row = n0 + r;
                const size_t idx = (row_offset + row) * B + b;
                dst_s[r] =
                    row < n ? (scales_fp16 ? static_cast<float>(half::from_bits(s_u16[idx])) : s_f32[idx]) : 0.0f;
            }
            dst += 128;
            std::memcpy(dst, wsum, 32 * sizeof(int32_t));
            dst += 128;
        }
    }
}

bool arch_quantize_symmetric_blocks_f32(int32_t bits, size_t m, size_t k, size_t block_size, float static_scale,
                                        const float* input, int8_t* quantized_input, float* input_scales,
                                        size_t block_start, size_t block_end) {
    if (!has_x86_vnni()) return false;
    const size_t B = k / block_size;
    const size_t padded_block = symmetric_linear_padded_block(block_size);
    quantize_symmetric_blocks_x86(bits, k, block_size, static_scale, input, reinterpret_cast<uint8_t*>(quantized_input),
                                  input_scales, block_start, block_end, B, padded_block);
    return true;
}

bool arch_symmetric_linear_f32(int32_t bits, size_t m, size_t n, size_t k, size_t block_size, float output_scale,
                               const void* packed_tiles, const int8_t* quantized_input, const float* input_scales,
                               float* output, size_t output_stride, size_t tile_start, size_t tile_end) {
    if (!has_x86_vnni()) return false;
    const size_t B = k / block_size;
    const size_t padded_block = symmetric_linear_padded_block(block_size);
    const auto* ptr = static_cast<const uint8_t*>(packed_tiles);
    const auto* act_u8 = reinterpret_cast<const uint8_t*>(quantized_input);
    if (bits == 8) {
        if (m >= 2 && padded_block >= 64 && has_x86_amx_int8()) {
            dispatch_i8_tiles_amx(m, n, B, B * padded_block, padded_block, output_scale, ptr, act_u8, input_scales,
                                  nullptr, output, output_stride, tile_start, tile_end);
        } else {
            dispatch_i8_tiles_vnni(m, n, B, B * padded_block, padded_block, output_scale, ptr, act_u8, input_scales,
                                   nullptr, output, output_stride, tile_start, tile_end);
        }
    } else {
        dispatch_symmetric_lowbit_x86(bits, m, n, B, padded_block, output_scale, ptr, act_u8, input_scales, output,
                                      output_stride, tile_start, tile_end);
    }
    return true;
}

} // namespace ynn::detail
