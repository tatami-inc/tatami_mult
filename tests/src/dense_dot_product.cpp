#include <gtest/gtest.h>

#include <algorithm>

#include "tatami_test/tatami_test.hpp"

#include "tatami_mult/dense_dot_product.hpp"

class DenseDotProductTest : public ::testing::TestWithParam<int> {};

TEST_P(DenseDotProductTest, Basic) {
    const auto N = GetParam();

    auto left = tatami_test::simulate_vector<double>(N, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.seed = 28 + N;
        return opt;
    }());
    auto right = tatami_test::simulate_vector<double>(N, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.seed = 51 + N;
        return opt;
    }());

    const auto x0 = std::inner_product(left.begin(), left.end(), right.data(), 0.0);
    const auto x1 = tatami_mult::dense_dot_product<1>(N, left.data(), right.data(), 0.0);
    const auto x2 = tatami_mult::dense_dot_product<4>(N, left.data(), right.data(), 0.0);
    EXPECT_FLOAT_EQ(x0, x1);
    EXPECT_FLOAT_EQ(x0, x2);

    // Check that initial value works as expected.
    const auto y0 = std::inner_product(left.begin(), left.end(), right.data(), 123.4);
    const auto y1 = tatami_mult::dense_dot_product<1>(N, left.data(), right.data(), 123.4);
    const auto y2 = tatami_mult::dense_dot_product<4>(N, left.data(), right.data(), 123.4);
    EXPECT_FLOAT_EQ(y0, y1);
    EXPECT_FLOAT_EQ(y0, y2);
}

INSTANTIATE_TEST_SUITE_P(
    DenseDotProduct,
    DenseDotProductTest,
    ::testing::Values(1, 3, 4, 5, 51, 120, 130, 141) // check multiples and non-multiples of 4.
);
