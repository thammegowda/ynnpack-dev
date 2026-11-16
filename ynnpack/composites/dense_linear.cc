// Licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/type.h"
#include "ynnpack/composites/composites.h"
#include "ynnpack/kernels/dot/dot.h"
#include "ynnpack/kernels/dot/pack.h"
#include "ynnpack/kernels/dot/schedule.h"
#include "ynnpack/kernels/transpose/transpose.h"

namespace ynn {
namespace {

constexpr size_t kCacheSizeL2 = 128 * 1024;

inline void f32_to_bf16_slice(const float* src, uint16_t* dst, size_t count) noexcept {
    size_t i = 0;
#if defined(__ARM_NEON) && (defined(__aarch64__) || defined(__arm64__)) && defined(__ARM_FEATURE_BF16)
    for (; i + 8 <= count; i += 8) {
        const float32x4_t v0 = vld1q_f32(src + i);
        const float32x4_t v1 = vld1q_f32(src + i + 4);
        vst1q_u16(dst + i, vreinterpretq_u16_bf16(vcombine_bf16(vcvt_bf16_f32(v0), vcvt_bf16_f32(v1))));
    }
#endif
    for (; i < count; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, src + i, sizeof(bits));
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        dst[i] = static_cast<uint16_t>(bits >> 16);
    }
}

inline void bf16_to_f32_slice(const uint16_t* src, float* dst, size_t count) noexcept {
    size_t i = 0;
#if defined(__ARM_NEON)
    for (; i + 8 <= count; i += 8) {
        const uint16x8_t v = vld1q_u16(src + i);
        const uint32x4_t lo = vshlq_n_u32(vmovl_u16(vget_low_u16(v)), 16);
        const uint32x4_t hi = vshlq_n_u32(vmovl_u16(vget_high_u16(v)), 16);
        vst1q_f32(dst + i, vreinterpretq_f32_u32(lo));
        vst1q_f32(dst + i + 4, vreinterpretq_f32_u32(hi));
    }
#endif
    for (; i < count; ++i) {
        const uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(dst + i, &bits, sizeof(float));
    }
}

inline dot_kernel lookup_dot_kernel(const dot_type& type, const dot_shape& shape, const dot_packed_shape& packed,
                                    uint32_t flags, std::optional<bool> transpose_a) noexcept {
    struct cached_lookup {
        dot_type type = {};
        dot_shape shape = {};
        dot_packed_shape packed = {};
        uint32_t flags = 0;
        int req_ta = -2;
        dot_kernel kernel = {};
    };
    static thread_local cached_lookup tls;
    const int req_ta = transpose_a ? (*transpose_a ? 1 : 0) : -1;
    if (tls.kernel.kernel && tls.type.a == type.a && tls.type.b == type.b && tls.type.c == type.c &&
        tls.shape.m == shape.m && tls.shape.n == shape.n && tls.shape.k1 == shape.k1 && tls.shape.k2 == shape.k2 &&
        tls.shape.k3 == shape.k3 && tls.packed.block_n == packed.block_n && tls.packed.tile_k == packed.tile_k &&
        tls.flags == flags && tls.req_ta == req_ta) {
        return tls.kernel;
    }
    const dot_kernel kernel = get_dot_kernel(type, shape, packed, flags, transpose_a);
    tls = {type, shape, packed, flags, req_ta, kernel};
    return kernel;
}

inline void run_dot_tile(const dot_kernel& kernel, size_t m, size_t n_valid, size_t k, size_t block_n, size_t tile_k,
                         size_t a_elem_size, size_t b_elem_size, bool transposed_a, const void* a_ptr,
                         size_t a_stride_m, size_t a_stride_k, const void* b_tile, const float* bias_tile,
                         float* c_tile, size_t c_stride_m, dot_kernel_state* state) noexcept {
    const size_t k1 = k & ~(tile_k - 1);
    const size_t k1_tail = k & (tile_k - 1);
    const size_t c_elem_size = sizeof(float);
    const size_t b_stride_k = block_n * b_elem_size;
    const size_t b_stride_n = tile_k * b_elem_size;
    const size_t cache_sizes[] = {kCacheSizeL2};
    dot_loop loops_storage[3];

    auto call_kernel = [&](size_t m_i, size_t n_i, span<const size_t> ks, const void* a, size_t a_m,
                           span<const size_t> a_ks, const void* b, span<const size_t> b_ks, size_t init_c_m,
                           const void* init_c, void* c, dot_kernel_state* st) {
        const size_t eff_a_m = (kernel.flags & dot_flag::transpose_a) ? a_ks[0] : a_m;
        kernel.kernel(m_i, n_i, ks[2], ks[1], ks[0], eff_a_m, a_ks[2], a_ks[1], a, b_ks[2], b_ks[1], b_ks[0], b,
                      init_c_m, init_c, c_stride_m, c, st);
    };

    if (k1 > 0) {
        const size_t ks[] = {k1, 1, 1};
        const size_t a_k_strides[] = {a_stride_k, 0, 0};
        const size_t b_k_strides[] = {b_stride_k, 0, 0};
        auto loops = schedule_dot(cache_sizes, m, n_valid, ks, kernel.block_m, kernel.block_n, kernel.block_k,
                                  a_elem_size, b_elem_size, loops_storage);
        run_dot(loops, m, n_valid, ks, kernel.block_m, kernel.block_n, kernel.block_k, a_stride_m, a_k_strides, a_ptr,
                b_k_strides, b_stride_n, b_tile, 0, bias_tile, c_stride_m, c_elem_size, c_tile, call_kernel, state);
    }

    if (k1_tail > 0) {
        const size_t ks_tail[] = {k1_tail, 1, 1};
        const size_t a_k_strides[] = {a_stride_k, 0, 0};
        const size_t b_k_strides[] = {b_stride_k, 0, 0};
        auto loops = schedule_dot(cache_sizes, m, n_valid, ks_tail, kernel.block_m, kernel.block_n, kernel.block_k,
                                  a_elem_size, b_elem_size, loops_storage);
        const size_t a_padded_stride = a_elem_size * tile_k;
        void* a_padded = YNN_ALLOCA(uint8_t, static_cast<size_t>(kernel.block_m) * a_padded_stride);
        std::memset(a_padded, 0, static_cast<size_t>(kernel.block_m) * a_padded_stride);

        auto call_tail = [&](size_t m_i, size_t n_i, span<const size_t> ks, const void* a, size_t a_m,
                             span<const size_t> /*a_ks*/, const void* b, span<const size_t> b_ks, size_t init_c_m,
                             const void* init_c, void* c, dot_kernel_state* st) {
            for (size_t row = 0; row < m_i; ++row) {
                std::memcpy(offset_bytes(a_padded, row * a_padded_stride), offset_bytes(a, row * a_m),
                            ks[0] * a_elem_size);
            }
            kernel.kernel(m_i, n_i, 1, 1, tile_k, a_padded_stride, 0, 0, a_padded, 0, 0, b_ks[0], b, init_c_m, init_c,
                          c_stride_m, c, st);
        };

        const void* tail_a = offset_bytes(a_ptr, a_stride_k * k1);
        const void* tail_b = offset_bytes(b_tile, b_stride_k * k1);
        const void* tail_init = k1 > 0 ? static_cast<const void*>(c_tile) : static_cast<const void*>(bias_tile);
        const size_t tail_init_stride = k1 > 0 ? c_stride_m : 0;
        run_dot(loops, m, n_valid, ks_tail, kernel.block_m, kernel.block_n, kernel.block_k, a_stride_m, a_k_strides,
                tail_a, b_k_strides, b_stride_n, tail_b, tail_init_stride, tail_init, c_stride_m, c_elem_size, c_tile,
                call_tail, state);
    }
}

inline void run_i32_dot_tile(const dot_kernel& kernel, size_t m, size_t n_valid, size_t k, size_t block_n,
                             size_t tile_k, bool transposed_a, const void* a_ptr, size_t a_stride_m, size_t a_stride_k,
                             const void* b_tile, int32_t* c_tile, size_t c_stride_m, dot_kernel_state* state) noexcept {
    const size_t k1 = k & ~(tile_k - 1);
    const size_t k1_tail = k & (tile_k - 1);
    const size_t c_elem_size = sizeof(int32_t);
    const size_t b_stride_k = block_n;
    const size_t b_stride_n = tile_k;
    const size_t cache_sizes[] = {kCacheSizeL2};
    dot_loop loops_storage[3];

    auto call_kernel = [&](size_t m_i, size_t n_i, span<const size_t> ks, const void* a, size_t a_m,
                           span<const size_t> a_ks, const void* b, span<const size_t> b_ks, size_t init_c_m,
                           const void* init_c, void* c, dot_kernel_state* st) {
        const size_t eff_a_m = (kernel.flags & dot_flag::transpose_a) ? a_ks[0] : a_m;
        kernel.kernel(m_i, n_i, ks[2], ks[1], ks[0], eff_a_m, a_ks[2], a_ks[1], a, b_ks[2], b_ks[1], b_ks[0], b,
                      init_c_m, init_c, c_stride_m, c, st);
    };

    if (k1 > 0) {
        const size_t ks[] = {k1, 1, 1};
        const size_t a_k_strides[] = {a_stride_k, 0, 0};
        const size_t b_k_strides[] = {b_stride_k, 0, 0};
        auto loops = schedule_dot(cache_sizes, m, n_valid, ks, kernel.block_m, kernel.block_n, kernel.block_k, 1, 1,
                                  loops_storage);
        run_dot(loops, m, n_valid, ks, kernel.block_m, kernel.block_n, kernel.block_k, a_stride_m, a_k_strides, a_ptr,
                b_k_strides, b_stride_n, b_tile, 0, nullptr, c_stride_m, c_elem_size, c_tile, call_kernel, state);
    }

    if (k1_tail > 0) {
        const size_t ks_tail[] = {k1_tail, 1, 1};
        const size_t a_k_strides[] = {a_stride_k, 0, 0};
        const size_t b_k_strides[] = {b_stride_k, 0, 0};
        auto loops = schedule_dot(cache_sizes, m, n_valid, ks_tail, kernel.block_m, kernel.block_n, kernel.block_k, 1,
                                  1, loops_storage);
        void* a_padded = YNN_ALLOCA(uint8_t, static_cast<size_t>(kernel.block_m) * tile_k);
        std::memset(a_padded, 0, static_cast<size_t>(kernel.block_m) * tile_k);

        auto call_tail = [&](size_t m_i, size_t n_i, span<const size_t> ks, const void* a, size_t a_m,
                             span<const size_t> /*a_ks*/, const void* b, span<const size_t> b_ks, size_t init_c_m,
                             const void* init_c, void* c, dot_kernel_state* st) {
            for (size_t row = 0; row < m_i; ++row) {
                std::memcpy(offset_bytes(a_padded, row * tile_k), offset_bytes(a, row * a_m), ks[0]);
            }
            kernel.kernel(m_i, n_i, 1, 1, tile_k, tile_k, 0, 0, a_padded, 0, 0, b_ks[0], b, init_c_m, init_c,
                          c_stride_m, c, st);
        };

        const void* tail_a = offset_bytes(a_ptr, a_stride_k * k1);
        const void* tail_b = offset_bytes(b_tile, b_stride_k * k1);
        const void* tail_init = k1 > 0 ? static_cast<const void*>(c_tile) : nullptr;
        const size_t tail_init_stride = k1 > 0 ? c_stride_m : 0;
        run_dot(loops, m, n_valid, ks_tail, kernel.block_m, kernel.block_n, kernel.block_k, a_stride_m, a_k_strides,
                tail_a, b_k_strides, b_stride_n, tail_b, tail_init_stride, tail_init, c_stride_m, c_elem_size, c_tile,
                call_tail, state);
    }
}

} // namespace

ynn_status make_dense_linear_plan(ynn_type a_type, ynn_type b_type, ynn_type c_type, size_t n, size_t k,
                                  dense_linear_plan* out_plan) {
    if (!out_plan || n == 0 || k == 0) return ynn_status_invalid_parameter;
    const dot_kernel k0 = get_dot_kernel({a_type, b_type, c_type}, {unknown_dot_extent, n, k});
    if (!k0.kernel || k0.tile_k <= 0 || k0.tile_n <= 0 || k0.max_block_n <= 0) {
        return ynn_status_unsupported_parameter;
    }

    const size_t tile_k = static_cast<size_t>(k0.tile_k);
    const size_t tile_n = static_cast<size_t>(k0.tile_n);
    const size_t elem_bits = type_size_bits(b_type);
    const size_t elem_bytes = type_size_bytes(b_type);
    const size_t align_block_n = std::max<size_t>(tile_n, static_cast<size_t>(k0.max_block_n));

    size_t block_n = align_block_n;
    if (elem_bytes > 1) {
        const size_t cache_elements = kCacheSizeL2 * 8 / elem_bits;
        const size_t cache_blocks_n = std::max<size_t>(1, cache_elements / (align_block_n * k));
        block_n = align_block_n * cache_blocks_n;
    }
    block_n = std::max<size_t>(tile_n, std::min<size_t>(align_up(n, tile_n), block_n));

    const size_t tiles_k = ceil_div(k, tile_k);
    const size_t blocks_n = ceil_div(n, block_n);
    const size_t tile_bytes = tiles_k * block_n * tile_k * elem_bytes;

    *out_plan = dense_linear_plan{
        .a_type = a_type,
        .b_type = b_type,
        .c_type = c_type,
        .n = n,
        .k = k,
        .block_n = block_n,
        .tile_k = tile_k,
        .tiles_k = tiles_k,
        .blocks_n = blocks_n,
        .tile_bytes = tile_bytes,
        .total_bytes = blocks_n * tile_bytes,
    };
    return ynn_status_success;
}

void pack_dense_linear_tiles(const dense_linear_plan& plan, bool transpose_b, const void* b_ptr,
                             size_t b_row_stride_bytes, size_t tile_start, size_t tile_end, void* packed_tiles,
                             int32_t* col_sums) {
    tile_end = std::min(tile_end, plan.blocks_n);
    if (tile_start >= tile_end) return;

    const size_t elem_bits = type_size_bits(plan.b_type);
    const size_t elem_bytes = type_size_bytes(plan.b_type);
    const size_t output_ko_stride = plan.block_n * plan.tile_k * elem_bytes;
    const bool use_transpose_packer = !transpose_b || get_transpose_kernel(elem_bits * plan.tile_k) != nullptr;
    packer pack_op(transpose_b, elem_bits, plan.tile_k, plan.block_n);

    for (size_t tile = tile_start; tile < tile_end; ++tile) {
        const size_t n0 = tile * plan.block_n;
        const size_t n_valid = std::min(plan.block_n, plan.n - n0);
        uint8_t* dst = static_cast<uint8_t*>(packed_tiles) + tile * plan.tile_bytes;
        if (n_valid < plan.block_n || (plan.k % plan.tile_k) != 0) {
            std::memset(dst, 0, plan.tile_bytes);
        }

        if (use_transpose_packer) {
            const void* src =
                transpose_b ? offset_bytes(b_ptr, n0 * b_row_stride_bytes) : offset_bytes(b_ptr, n0 * elem_bytes);
            pack_op.pack(plan.k, n_valid, b_row_stride_bytes, src, output_ko_stride, 0, dst);
        } else {
            // Generic scalar fallback if transpose_kernel does not exist for wide tile_k.
            const auto* src_bytes = static_cast<const uint8_t*>(b_ptr);
            for (size_t r = 0; r < n_valid; ++r) {
                const uint8_t* row = src_bytes + (n0 + r) * b_row_stride_bytes;
                for (size_t kk = 0; kk < plan.k; ++kk) {
                    const size_t ko = kk / plan.tile_k;
                    const size_t ki = kk % plan.tile_k;
                    std::memcpy(dst + ko * output_ko_stride + (r * plan.tile_k + ki) * elem_bytes,
                                row + kk * elem_bytes, elem_bytes);
                }
            }
        }

        if (col_sums != nullptr && plan.b_type == ynn_type_int8) {
            const auto* src_i8 = static_cast<const int8_t*>(b_ptr);
            if (transpose_b) {
                for (size_t r = 0; r < n_valid; ++r) {
                    const int8_t* row =
                        reinterpret_cast<const int8_t*>(offset_bytes(src_i8, (n0 + r) * b_row_stride_bytes));
                    int32_t sum = 0;
                    for (size_t kk = 0; kk < plan.k; ++kk) sum += row[kk];
                    col_sums[n0 + r] = sum;
                }
            } else {
                std::fill_n(col_sums + n0, n_valid, int32_t{0});
                for (size_t kk = 0; kk < plan.k; ++kk) {
                    const int8_t* row =
                        reinterpret_cast<const int8_t*>(offset_bytes(src_i8, kk * b_row_stride_bytes)) + n0;
                    for (size_t r = 0; r < n_valid; ++r) col_sums[n0 + r] += row[r];
                }
            }
        }
    }
}

bool dense_linear_select_kernel(const dense_linear_plan& plan, size_t m, bool* out_transpose_a, size_t* out_tile_m,
                                size_t* out_a_packed_bytes) {
    const std::optional<bool> req_ta = m == 1 ? std::nullopt : std::optional<bool>{};
    const dot_kernel kernel =
        lookup_dot_kernel({plan.a_type, plan.b_type, plan.c_type}, {m, plan.n, plan.k, 1, 1},
                          {static_cast<int>(plan.block_n), static_cast<int>(plan.tile_k)}, 0, req_ta);
    if (!kernel.kernel) return false;
    const bool ta = m > 1 && (kernel.flags & dot_flag::transpose_a) != 0;
    const size_t tile_m = static_cast<size_t>(kernel.tile_m);
    const size_t m_aligned = align_up(m, tile_m);
    *out_transpose_a = ta;
    *out_tile_m = tile_m;
    *out_a_packed_bytes = ta ? plan.tiles_k * m_aligned * plan.tile_k * type_size_bytes(plan.a_type) : 0;
    return true;
}

void dense_linear_pack_a(const dense_linear_plan& plan, size_t m, size_t tile_m, const void* a_ptr,
                         size_t a_row_stride_bytes, void* a_packed) {
    const size_t elem_bits = type_size_bits(plan.a_type);
    const size_t elem_bytes = type_size_bytes(plan.a_type);
    const size_t m_aligned = align_up(m, tile_m);
    const size_t ko_stride = m_aligned * plan.tile_k * elem_bytes;
    packer pack_a(true, elem_bits, plan.tile_k, m_aligned * plan.tile_k);
    pack_a.pack(plan.k, m, a_row_stride_bytes, a_ptr, ko_stride, 0, a_packed);
}

void dense_linear_run_tiles(const dense_linear_plan& plan, size_t m, bool transposed_a, size_t tile_m,
                            const void* a_ptr, size_t a_row_stride_bytes, const void* packed_b, const float* bias_f32,
                            float* output_f32, size_t output_stride_elems, size_t tile_start, size_t tile_end) {
    tile_end = std::min(tile_end, plan.blocks_n);
    if (tile_start >= tile_end || m == 0) return;

    const std::optional<bool> req_ta = m == 1 ? std::nullopt : std::optional<bool>{transposed_a};
    const dot_kernel kernel =
        lookup_dot_kernel({plan.a_type, plan.b_type, plan.c_type}, {m, plan.n, plan.k, 1, 1},
                          {static_cast<int>(plan.block_n), static_cast<int>(plan.tile_k)}, 0, req_ta);
    assert(kernel.kernel);

    const size_t a_elem_size = type_size_bytes(plan.a_type);
    const size_t b_elem_size = type_size_bytes(plan.b_type);
    const size_t m_aligned = align_up(m, tile_m);
    const size_t eff_a_stride_m = transposed_a ? plan.tile_k * a_elem_size : a_row_stride_bytes;
    const size_t eff_a_stride_k = transposed_a ? m_aligned * a_elem_size : a_elem_size;
    const size_t c_stride_bytes = output_stride_elems * sizeof(float);

    dot_kernel_state kernel_state = {};
    for (size_t tile = tile_start; tile < tile_end; ++tile) {
        const size_t n0 = tile * plan.block_n;
        const size_t n_valid = std::min(plan.block_n, plan.n - n0);
        const void* b_tile = static_cast<const uint8_t*>(packed_b) + tile * plan.tile_bytes;
        const float* bias_tile = bias_f32 != nullptr ? bias_f32 + n0 : nullptr;
        float* c_tile = output_f32 + n0;
        run_dot_tile(kernel, m, n_valid, plan.k, plan.block_n, plan.tile_k, a_elem_size, b_elem_size, transposed_a,
                     a_ptr, eff_a_stride_m, eff_a_stride_k, b_tile, bias_tile, c_tile, c_stride_bytes, &kernel_state);
    }
}

void quantize_dynamic_int8_rows_f32(size_t rows, size_t k, const float* input, int8_t* quantized, float* scales,
                                    int32_t* zero_points, size_t row_start, size_t row_end) {
    row_end = std::min(row_end, rows);
    for (size_t row = row_start; row < row_end; ++row) {
        const float* x = input + row * k;
        float rmin = 0.0f;
        float rmax = 0.0f;
        size_t col = 0;
#if defined(__ARM_NEON)
        float32x4_t vmin0 = vdupq_n_f32(0.0f), vmin1 = vdupq_n_f32(0.0f);
        float32x4_t vmax0 = vdupq_n_f32(0.0f), vmax1 = vdupq_n_f32(0.0f);
        for (; col + 8 <= k; col += 8) {
            const float32x4_t x0 = vld1q_f32(x + col);
            const float32x4_t x1 = vld1q_f32(x + col + 4);
            vmin0 = vminq_f32(vmin0, x0);
            vmin1 = vminq_f32(vmin1, x1);
            vmax0 = vmaxq_f32(vmax0, x0);
            vmax1 = vmaxq_f32(vmax1, x1);
        }
        rmin = vminvq_f32(vminq_f32(vmin0, vmin1));
        rmax = vmaxvq_f32(vmaxq_f32(vmax0, vmax1));
#endif
        for (; col < k; ++col) {
            rmin = std::min(rmin, x[col]);
            rmax = std::max(rmax, x[col]);
        }
        const float rdiff = rmax - rmin;
        float scale = 1.0f;
        float inv_scale = 0.0f;
        int32_t zp = -128;
        if (rdiff > 0.0f) {
            inv_scale = 255.0f / rdiff;
            scale = rdiff / 255.0f;
            const float min_error = -128.0f + rmin * inv_scale;
            const float max_error = 127.0f + rmax * inv_scale;
            const float fzp = min_error + max_error > 0.0f ? -128.0f - rmin * inv_scale : 127.0f - rmax * inv_scale;
            zp = static_cast<int32_t>(std::rint(std::clamp(fzp, -128.0f, 127.0f)));
        }
        scales[row] = scale;
        zero_points[row] = zp;
        int8_t* q = quantized + row * k;
        col = 0;
#if defined(__ARM_NEON) && (defined(__aarch64__) || defined(__arm64__))
        const float32x4_t vinv = vdupq_n_f32(inv_scale);
        const int32x4_t vzp = vdupq_n_s32(zp);
        for (; col + 16 <= k; col += 16) {
            const int32x4_t i0 = vaddq_s32(vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x + col + 0), vinv)), vzp);
            const int32x4_t i1 = vaddq_s32(vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x + col + 4), vinv)), vzp);
            const int32x4_t i2 = vaddq_s32(vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x + col + 8), vinv)), vzp);
            const int32x4_t i3 = vaddq_s32(vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x + col + 12), vinv)), vzp);
            const int16x8_t s0 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
            const int16x8_t s1 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
            vst1q_s8(q + col, vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1)));
        }
#endif
        for (; col < k; ++col) {
            const int32_t v = static_cast<int32_t>(std::lround(x[col] * inv_scale)) + zp;
            q[col] = static_cast<int8_t>(std::clamp<int32_t>(v, -128, 127));
        }
    }
}

size_t quantized_linear_scratch_bytes(const dense_linear_plan& plan, size_t m, size_t tile_m) {
    const size_t m_step = std::min(m, align_up(size_t{32}, std::max<size_t>(1, tile_m)));
    return std::max<size_t>(1, m_step) * plan.block_n * sizeof(int32_t);
}

void quantized_linear_run_tiles(const dense_linear_plan& plan, size_t m, bool transposed_a, size_t tile_m,
                                const void* a_ptr, size_t a_row_stride_bytes, const void* packed_b,
                                const int32_t* b_col_sums, const float* a_scales, const int32_t* a_zero_points,
                                const float* b_scales, const float* bias_f32, float* output_f32,
                                size_t output_stride_elems, int32_t* i32_scratch, size_t tile_start, size_t tile_end) {
    tile_end = std::min(tile_end, plan.blocks_n);
    if (tile_start >= tile_end || m == 0) return;

    const std::optional<bool> req_ta = m == 1 ? std::nullopt : std::optional<bool>{transposed_a};
    const dot_kernel kernel =
        lookup_dot_kernel({ynn_type_int8, ynn_type_int8, ynn_type_int32}, {m, plan.n, plan.k, 1, 1},
                          {static_cast<int>(plan.block_n), static_cast<int>(plan.tile_k)}, 0, req_ta);
    assert(kernel.kernel);

    const size_t m_aligned = align_up(m, tile_m);
    const size_t eff_a_stride_m = transposed_a ? plan.tile_k : a_row_stride_bytes;
    const size_t eff_a_stride_k = transposed_a ? m_aligned : 1;
    const size_t c_i32_stride_bytes = plan.block_n * sizeof(int32_t);
    const size_t m_step = align_up(size_t{32}, std::max<size_t>(1, tile_m));

    dot_kernel_state kernel_state = {};
    for (size_t tile = tile_start; tile < tile_end; ++tile) {
        const size_t n0 = tile * plan.block_n;
        const size_t n_valid = std::min(plan.block_n, plan.n - n0);
        const void* b_tile = static_cast<const uint8_t*>(packed_b) + tile * plan.tile_bytes;
        const float* ws = b_scales + n0;
        const int32_t* wsum = b_col_sums + n0;
        const float* bp = bias_f32 != nullptr ? bias_f32 + n0 : nullptr;

        for (size_t m0 = 0; m0 < m; m0 += m_step) {
            const size_t m_count = std::min(m_step, m - m0);
            const void* a_chunk = offset_bytes(a_ptr, m0 * eff_a_stride_m);
            run_i32_dot_tile(kernel, m_count, n_valid, plan.k, plan.block_n, plan.tile_k, transposed_a, a_chunk,
                             eff_a_stride_m, eff_a_stride_k, b_tile, i32_scratch, c_i32_stride_bytes, &kernel_state);

            for (size_t row = 0; row < m_count; ++row) {
                const float a_scale = a_scales[m0 + row];
                const int32_t a_zp = a_zero_points[m0 + row];
                const int32_t* acc_row = i32_scratch + row * plan.block_n;
                float* out_row = output_f32 + (m0 + row) * output_stride_elems + n0;
                size_t col = 0;
#if defined(__ARM_NEON)
                const float32x4_t vas = vdupq_n_f32(a_scale);
                const int32x4_t vazp = vdupq_n_s32(a_zp);
                for (; col + 4 <= n_valid; col += 4) {
                    const int32x4_t corr = vmlsq_s32(vld1q_s32(acc_row + col), vld1q_s32(wsum + col), vazp);
                    const float32x4_t prod = vmulq_f32(vcvtq_f32_s32(corr), vmulq_f32(vld1q_f32(ws + col), vas));
                    const float32x4_t res = bp ? vaddq_f32(prod, vld1q_f32(bp + col)) : prod;
                    vst1q_f32(out_row + col, res);
                }
#endif
                for (; col < n_valid; ++col) {
                    const int32_t corr = acc_row[col] - a_zp * wsum[col];
                    const float val = static_cast<float>(corr) * (a_scale * ws[col]);
                    out_row[col] = bp ? (val + bp[col]) : val;
                }
            }
        }
    }
}

void convert_f32_to_bf16_rows(size_t rows, size_t cols, const float* src, size_t src_stride_elems, uint16_t* dst,
                              size_t dst_stride_elems, size_t row_start, size_t row_end) {
    row_end = std::min(row_end, rows);
    for (size_t row = row_start; row < row_end; ++row) {
        f32_to_bf16_slice(src + row * src_stride_elems, dst + row * dst_stride_elems, cols);
    }
}

void convert_bf16_to_f32_rows(size_t rows, size_t cols, const uint16_t* src, size_t src_stride_elems, float* dst,
                              size_t dst_stride_elems, size_t row_start, size_t row_end) {
    row_end = std::min(row_end, rows);
    for (size_t row = row_start; row < row_end; ++row) {
        bf16_to_f32_slice(src + row * src_stride_elems, dst + row * dst_stride_elems, cols);
    }
}

} // namespace ynn
