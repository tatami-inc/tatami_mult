#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "tatami_test/tatami_test.hpp"

#include "tatami_mult/single_vector/sparse_row.hpp"

class SingleVectorSparseRowTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {};

TEST_P(SingleVectorSparseRowTest, Vector) {
    const auto params = GetParam();
    const int NR = std::get<0>(params);
    const int NC = std::get<1>(params);
    const auto nthreads = std::get<2>(params);

    auto dump = tatami_test::simulate_vector<double>(NR * NC, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.density = 0.21;
        opt.seed = 69 + NR + NC + nthreads;
        return opt;
    }());
    auto sparse_row = tatami::convert_to_compressed_sparse<double, int>(tatami::DenseRowMatrix<double, int>(NR, NC, dump), true, {});
    auto sparse_col = tatami::convert_to_compressed_sparse<double, int>(*sparse_row, false, {});

    auto rhs = tatami_test::simulate_vector<double>(NC, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 43 + NR + NC + nthreads;
        return opt;
    }());

    tatami_mult::MultiplySparseRowWithSingleVectorOptions opt;
    opt.num_threads = nthreads;

    {
        std::vector<double> output1(NR);
        tatami_mult::multiply_sparse_row_with_single_vector<1>(*sparse_row, rhs.data(), output1.data(), opt);
        std::vector<double> output4(NR);
        tatami_mult::multiply_sparse_row_with_single_vector<4>(*sparse_row, rhs.data(), output4.data(), opt);

        for (int r = 0; r < NR; ++r) {
            const auto ref = std::inner_product(rhs.begin(), rhs.end(), dump.begin() + r * NC, 0.0);
            EXPECT_FLOAT_EQ(ref, output1[r]);
            EXPECT_FLOAT_EQ(ref, output4[r]);
        }
    }

    {
        std::vector<double> output1(NR);
        tatami_mult::multiply_sparse_row_with_single_vector<1>(*sparse_col, rhs.data(), output1.data(), opt);
        std::vector<double> output4(NR);
        tatami_mult::multiply_sparse_row_with_single_vector<4>(*sparse_col, rhs.data(), output4.data(), opt);

        for (int r = 0; r < NR; ++r) {
            const auto ref = std::inner_product(rhs.begin(), rhs.end(), dump.begin() + r * NC, 0.0);
            EXPECT_FLOAT_EQ(ref, output1[r]);
            EXPECT_FLOAT_EQ(ref, output4[r]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    SingleVector,
    SingleVectorSparseRowTest,
    ::testing::Combine(
        ::testing::Values(107, 24),
        ::testing::Values(51, 138),
        ::testing::Values(1, 3)
    )
);
