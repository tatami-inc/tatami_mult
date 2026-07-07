#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "tatami_test/tatami_test.hpp"

#include "tatami_mult/single_vector/dense_row.hpp"

class SingleVectorDenseRowTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {};

TEST_P(SingleVectorDenseRowTest, Vector) {
    const auto params = GetParam();
    const int NR = std::get<0>(params);
    const int NC = std::get<1>(params);
    const auto nthreads = std::get<2>(params);

    auto dump = tatami_test::simulate_vector<double>(NR * NC, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 69 + nthreads;
        return opt;
    }());
    auto dense_row = std::make_unique<tatami::DenseRowMatrix<double, int> >(NR, NC, dump);
    auto dense_col = tatami::convert_to_dense<double, int>(*dense_row, false, {});

    auto rhs = tatami_test::simulate_vector<double>(NC, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 42 + nthreads;
        return opt;
    }());

    tatami_mult::MultiplyDenseRowWithSingleVectorOptions opt;
    opt.num_threads = nthreads;

    {
        std::vector<double> output1(NR);
        tatami_mult::multiply_dense_row_with_single_vector<1>(*dense_row, rhs.data(), output1.data(), opt);
        std::vector<double> output4(NR);
        tatami_mult::multiply_dense_row_with_single_vector<4>(*dense_row, rhs.data(), output4.data(), opt);

        for (int r = 0; r < NR; ++r) {
            const auto ref = std::inner_product(rhs.begin(), rhs.end(), dump.begin() + r * NC, 0.0);
            EXPECT_FLOAT_EQ(ref, output1[r]);
            EXPECT_FLOAT_EQ(ref, output4[r]);
        }
    }

    {
        std::vector<double> output1(NR);
        tatami_mult::multiply_dense_row_with_single_vector<1>(*dense_col, rhs.data(), output1.data(), opt);
        std::vector<double> output4(NR);
        tatami_mult::multiply_dense_row_with_single_vector<4>(*dense_col, rhs.data(), output4.data(), opt);

        for (int r = 0; r < NR; ++r) {
            const auto ref = std::inner_product(rhs.begin(), rhs.end(), dump.begin() + r * NC, 0.0);
            EXPECT_FLOAT_EQ(ref, output1[r]);
            EXPECT_FLOAT_EQ(ref, output4[r]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    SingleVector,
    SingleVectorDenseRowTest,
    ::testing::Combine(
        ::testing::Values(102, 33),
        ::testing::Values(59, 148),
        ::testing::Values(1, 3)
    )
);
