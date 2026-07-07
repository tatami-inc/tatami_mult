#include <gtest/gtest.h>

#include <numeric>
#include <algorithm>
#include <vector>
#include <random>

#include "tatami_test/tatami_test.hpp"

#include "tatami_mult/sparse_dot_product.hpp"

class SparseDotProductTest : public ::testing::TestWithParam<int> {};

TEST_P(SparseDotProductTest, Basic) {
    const auto nnz = GetParam();

    auto left_value = tatami_test::simulate_vector<double>(nnz, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.seed = 43 + nnz;
        return opt;
    }());

    const int N = 100 + nnz;
    std::vector<int> left_index(N);
    std::iota(left_index.begin(), left_index.end(), 0);
    std::mt19937_64 rng(/* seed = */ 69 + nnz);
    std::shuffle(left_index.begin(), left_index.end(), rng);
    left_index.resize(N);
    std::sort(left_index.begin(), left_index.end());

    std::vector<double> refleft(N);
    for (int i = 0; i < nnz; ++i) {
        refleft[left_index[i]] = left_value[i];
    }

    auto right = tatami_test::simulate_vector<double>(N, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.seed = 1187 + nnz;
        return opt;
    }());

    const auto x0 = std::inner_product(refleft.begin(), refleft.end(), right.data(), 0.0);
    const auto x1 = tatami_mult::sparse_dot_product<1>(nnz, left_value.data(), left_index.data(), right.data(), 0.0);
    const auto x2 = tatami_mult::sparse_dot_product<4>(nnz, left_value.data(), left_index.data(), right.data(), 0.0);
    EXPECT_FLOAT_EQ(x0, x1);
    EXPECT_FLOAT_EQ(x0, x2);

    // Check that initial value works as expected.
    const auto y0 = std::inner_product(refleft.begin(), refleft.end(), right.data(), 89.1);
    const auto y1 = tatami_mult::sparse_dot_product<1>(nnz, left_value.data(), left_index.data(), right.data(), 89.1);
    const auto y2 = tatami_mult::sparse_dot_product<4>(nnz, left_value.data(), left_index.data(), right.data(), 89.1);
    EXPECT_FLOAT_EQ(y0, y1);
    EXPECT_FLOAT_EQ(y0, y2);
}

INSTANTIATE_TEST_SUITE_P(
    SparseDotProduct,
    SparseDotProductTest,
    ::testing::Values(1, 3, 4, 5, 51, 120, 130, 141) // check multiples and non-multiples of 4.
);
