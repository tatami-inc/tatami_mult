#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "tatami_test/tatami_test.hpp"

#include "tatami_mult/single_vector/sparse_column.hpp"

class SingleVectorSparseColumnTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {};

TEST_P(SingleVectorSparseColumnTest, Vector) {
    const auto params = GetParam();
    const int NR = std::get<0>(params);
    const int NC = std::get<1>(params);
    const auto nthreads = std::get<2>(params);

    auto dump = tatami_test::simulate_vector<double>(NR * NC, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.density = 0.1;
        opt.seed = 69 + nthreads;
        return opt;
    }());
    auto sparse_row = tatami::convert_to_compressed_sparse<double, int>(tatami::DenseRowMatrix<double, int>(NR, NC, dump), true, {});
    auto sparse_col = tatami::convert_to_compressed_sparse<double, int>(*sparse_row, false, {});

    auto rhs = tatami_test::simulate_vector<double>(NC, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 42 + nthreads;
        return opt;
    }());

    tatami_mult::MultiplySparseColumnWithSingleVectorOptions opt;
    opt.num_threads = nthreads;

    {
        std::vector<double> output1(NR);
        tatami_mult::multiply_sparse_column_with_single_vector(*sparse_row, rhs.data(), output1.data(), opt);
        std::vector<double> output4(NR);
        tatami_mult::multiply_sparse_column_with_single_vector(*sparse_row, rhs.data(), output4.data(), opt);

        for (int r = 0; r < NR; ++r) {
            const auto ref = std::inner_product(rhs.begin(), rhs.end(), dump.begin() + r * NC, 0.0);
            EXPECT_FLOAT_EQ(ref, output1[r]);
            EXPECT_FLOAT_EQ(ref, output4[r]);
        }
    }

    {
        std::vector<double> output1(NR);
        tatami_mult::multiply_sparse_column_with_single_vector(*sparse_col, rhs.data(), output1.data(), opt);
        std::vector<double> output4(NR);
        tatami_mult::multiply_sparse_column_with_single_vector(*sparse_col, rhs.data(), output4.data(), opt);

        for (int r = 0; r < NR; ++r) {
            const auto ref = std::inner_product(rhs.begin(), rhs.end(), dump.begin() + r * NC, 0.0);
            EXPECT_FLOAT_EQ(ref, output1[r]);
            EXPECT_FLOAT_EQ(ref, output4[r]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    SingleVector,
    SingleVectorSparseColumnTest,
    ::testing::Combine(
        ::testing::Values(99, 53),
        ::testing::Values(72, 169),
        ::testing::Values(1, 3)
    )
);
