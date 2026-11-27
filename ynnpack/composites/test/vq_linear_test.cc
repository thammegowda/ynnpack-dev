#include "ynnpack/composites/vq_linear.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace ynn {
namespace {

struct Packed {
    std::vector<uint8_t> values, scales;
    std::vector<float> global;
};

std::vector<float> gaussian(size_t count, uint32_t seed) {
    std::mt19937 random(seed);
    std::normal_distribution<float> normal;
    std::vector<float> values(count);
    for (float& value : values) value = normal(random);
    return values;
}

Packed pack(int32_t dim, size_t rows, size_t cols, const std::vector<float>& weights, size_t row_block) {
    const size_t k_pad = vq_padded_cols(dim, cols);
    Packed packed{std::vector<uint8_t>(rows * (k_pad / dim)), std::vector<uint8_t>(rows * (k_pad / vq_group_size(dim))),
                  std::vector<float>(rows)};
    for (size_t begin = 0; begin < rows; begin += row_block)
        vq_pack_row_block_f32(dim, rows, cols, begin, begin + row_block, weights.data() + begin * cols,
                              packed.values.data(), packed.scales.data(), packed.global.data());
    return packed;
}

// Normalized Sylvester-Hadamard matrix entry, the transform the fast butterflies implement.
float hadamard(size_t i, size_t j, size_t n) {
    return (std::popcount(i & j) % 2 ? -1.0F : 1.0F) / std::sqrt(static_cast<float>(n));
}

TEST(VqLinearTest, HadamardTransformsMatchSylvesterMatrix) {
    for (const size_t block : {size_t{32}, size_t{64}}) {
        SCOPED_TRACE(block);
        constexpr size_t rows = 3;
        const size_t cols = 2 * block;
        const auto input = gaussian(rows * cols, 1);
        auto by_rows = input;
        hadamard_transform_rows_f32(by_rows.data(), rows, cols, block);
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < cols; ++c) {
                const size_t base = c / block * block;
                double expected = 0;
                for (size_t j = 0; j < block; ++j)
                    expected += hadamard(c - base, j, block) * input[r * cols + base + j];
                ASSERT_NEAR(by_rows[r * cols + c], expected, 1e-5) << "row " << r << " column " << c;
            }
    }
    constexpr size_t rows = 128, cols = 5, block = 64;
    const auto input = gaussian(rows * cols, 2);
    auto by_columns = input;
    hadamard_transform_cols_f32(by_columns.data(), rows, cols, block);
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c) {
            const size_t base = r / block * block;
            double expected = 0;
            for (size_t j = 0; j < block; ++j) expected += hadamard(r - base, j, block) * input[(base + j) * cols + c];
            ASSERT_NEAR(by_columns[r * cols + c], expected, 1e-5) << "row " << r << " column " << c;
        }
}

TEST(VqLinearTest, RowBlockPackingMatchesWholeMatrix) {
    constexpr size_t rows = 128, cols = 256;
    const auto weights = gaussian(rows * cols, 3);
    for (const int32_t dim : {3, 4}) {
        SCOPED_TRACE(dim);
        const auto whole = pack(dim, rows, cols, weights, rows);
        const auto blocks = pack(dim, rows, cols, weights, 64);
        EXPECT_EQ(blocks.values, whole.values);
        EXPECT_EQ(blocks.scales, whole.scales);
        EXPECT_EQ(blocks.global, whole.global);
    }
}

TEST(VqLinearTest, RotatedProductMatchesDecodedCodes) {
    constexpr size_t rows = 128, cols = 256, tokens = 3, row_begin = 32, row_end = 96;
    const auto weights = gaussian(rows * cols, 4);
    for (const int32_t dim : {3, 4}) {
        SCOPED_TRACE(dim);
        const size_t group = vq_group_size(dim), k_pad = vq_padded_cols(dim, cols), groups = k_pad / group;
        const auto packed = pack(dim, rows, cols, weights, rows);
        const auto codebook = dim == 3 ? vq3d_codebook() : vq4d_codebook();
        auto x_rot = gaussian(tokens * k_pad, 5);
        for (size_t t = 0; t < tokens; ++t)
            for (size_t k = cols; k < k_pad; ++k) x_rot[t * k_pad + k] = 0.0F;

        std::vector<double> expected(tokens * rows);
        for (size_t r = 0; r < rows; ++r)
            for (size_t g = 0; g < groups; ++g) {
                const double scale = double{packed.global[r]} * packed.scales[r * groups + g];
                for (size_t v = 0; v < 16; ++v) {
                    const uint8_t code = packed.values[r * (k_pad / dim) + g * 16 + v];
                    const double sign = code & 0x80 ? -1.0 : 1.0;
                    for (int32_t j = 0; j < dim; ++j) {
                        const double weight = scale * sign * codebook[code & 0x7F][j];
                        for (size_t t = 0; t < tokens; ++t)
                            expected[t * rows + r] += weight * x_rot[t * k_pad + g * group + v * dim + j];
                    }
                }
            }

        std::vector<float> actual(tokens * rows);
        vq_linear_rotated_f32(dim, tokens, rows, cols, x_rot.data(), packed.values.data(), packed.scales.data(),
                              packed.global.data(), actual.data(), 0, rows);
        for (size_t i = 0; i < actual.size(); ++i)
            ASSERT_NEAR(actual[i], expected[i], 1e-4 * (1.0 + std::abs(expected[i]))) << "at " << i;

        // A row range writes only its own rows.
        constexpr float untouched = -12345.0F;
        std::vector<float> partial(tokens * rows, untouched);
        vq_linear_rotated_f32(dim, tokens, rows, cols, x_rot.data(), packed.values.data(), packed.scales.data(),
                              packed.global.data(), partial.data(), row_begin, row_end);
        for (size_t t = 0; t < tokens; ++t)
            for (size_t r = 0; r < rows; ++r) {
                const size_t i = t * rows + r;
                if (r >= row_begin && r < row_end)
                    ASSERT_EQ(partial[i], actual[i]) << "at " << i;
                else
                    ASSERT_EQ(partial[i], untouched) << "at " << i;
            }
    }
}

TEST(VqLinearTest, ProjectionTracksDenseWeights) {
    constexpr size_t tokens = 4;
    // 128x256 rotates rows and columns (and pads VQ3D columns to 288); 96x192 skips the row rotation.
    for (const auto [rows, cols] : {std::pair<size_t, size_t>{128, 256}, std::pair<size_t, size_t>{96, 192}})
        for (const auto [dim, bound] : {std::pair{3, 0.24}, std::pair{4, 0.35}}) {
            SCOPED_TRACE(testing::Message() << rows << "x" << cols << " dim " << dim);
            const auto weights = gaussian(rows * cols, 6);
            const auto x = gaussian(tokens * cols, 7);
            const size_t k_pad = vq_padded_cols(dim, cols);
            const auto packed = pack(dim, rows, cols, weights, rows % 64 == 0 ? 64 : rows);

            std::vector<float> x_rot(tokens * k_pad, 0.0F);
            for (size_t t = 0; t < tokens; ++t) {
                std::copy_n(x.data() + t * cols, cols, x_rot.data() + t * k_pad);
                hadamard_transform_rows_f32(x_rot.data() + t * k_pad, 1, cols, vq_col_hadamard_block(cols));
            }
            std::vector<float> y(tokens * rows);
            vq_linear_rotated_f32(dim, tokens, rows, cols, x_rot.data(), packed.values.data(), packed.scales.data(),
                                  packed.global.data(), y.data(), 0, rows);
            hadamard_transform_rows_f32(y.data(), tokens, rows, vq_row_hadamard_block(rows));

            double error = 0, norm = 0;
            for (size_t t = 0; t < tokens; ++t)
                for (size_t r = 0; r < rows; ++r) {
                    double dense = 0;
                    for (size_t c = 0; c < cols; ++c) dense += double{weights[r * cols + c]} * x[t * cols + c];
                    error += std::pow(y[t * rows + r] - dense, 2);
                    norm += dense * dense;
                }
            EXPECT_LT(std::sqrt(error / norm), bound);
        }
}

} // namespace
} // namespace ynn
