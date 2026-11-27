#ifndef XNNPACK_YNNPACK_COMPOSITES_VQ_LINEAR_H_
#define XNNPACK_YNNPACK_COMPOSITES_VQ_LINEAR_H_

#include <cstddef>
#include <cstdint>

namespace ynn {

// Universal 128-entry antipodal spherical Gaussian codebooks (bit 7 = sign, bits 0..6 = index).
// Each centroid is stored as 4 floats (lane 3 is 0 for VQ3D) for aligned SIMD and Metal half4 loads.
// vq_linear.cc records their provenance; vq_codebook.py fits new tables and verifies these.
const float (*vq3d_codebook())[4];
const float (*vq4d_codebook())[4];

constexpr size_t vq_group_size(int32_t dim) { return dim == 3 ? 48 : 64; }
constexpr size_t vq_padded_cols(int32_t dim, size_t cols) {
    const size_t group = vq_group_size(dim);
    return ((cols + group - 1) / group) * group;
}
constexpr size_t vq_col_hadamard_block(size_t cols) { return cols % 64 == 0 ? 64 : (cols % 32 == 0 ? 32 : 0); }
constexpr size_t vq_row_hadamard_block(size_t rows) { return rows % 64 == 0 ? 64 : 0; }

// In-place normalized Walsh-Hadamard transform (H_block / sqrt(block)) along each row (contiguous stride `cols`).
void hadamard_transform_rows_f32(float* data, size_t rows, size_t cols, size_t block);

// In-place normalized Walsh-Hadamard transform along columns in contiguous 64-row blocks.
void hadamard_transform_cols_f32(float* data, size_t rows, size_t cols, size_t block);

// Packs rows [row_begin, row_end) of a dense F32 weight matrix [total_rows, cols] into VQ3D (dim=3, g=48) or
// VQ4D (dim=4, g=64) with 2D Hadamard rotation, variance-preserving scales, and two-level UINT8/FP32 scales.
// When total_rows % 64 == 0, row_begin and row_end must be multiples of 64.
// `weights_row_slice` points to row `row_begin` of the source matrix (stride `cols`).
void vq_pack_row_block_f32(int32_t dim, size_t total_rows, size_t cols, size_t row_begin, size_t row_end,
                           const float* weights_row_slice, uint8_t* out_values, uint8_t* out_scales, float* out_global);

// Computes y_rot[t, r] = global[r] * sum_g scales[r, g] * dot(x_rot[t, g], decode(values[r, g]))
// for r in [row_begin, row_end) and t in [0, tokens).
// `x_rot` has shape [tokens, vq_padded_cols(dim, cols)] (already column-Hadamard rotated and zero-padded).
// `y_rot` has shape [tokens, rows].
void vq_linear_rotated_f32(int32_t dim, size_t tokens, size_t rows, size_t cols, const float* x_rot,
                           const uint8_t* values, const uint8_t* scales, const float* global, float* y_rot,
                           size_t row_begin, size_t row_end);

} // namespace ynn

#endif // XNNPACK_YNNPACK_COMPOSITES_VQ_LINEAR_H_
