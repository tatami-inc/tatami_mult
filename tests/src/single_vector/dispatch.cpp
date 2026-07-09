#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "tatami_test/tatami_test.hpp"

#include "tatami_mult/single_vector/dispatch.hpp"

class SingleVectorDispatchTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {};

TEST_P(SingleVectorDispatchTest, Vector) {
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
    auto dense_row = std::make_unique<tatami::DenseRowMatrix<double, int> >(NR, NC, dump);
    auto dense_col = tatami::convert_to_dense<double, int>(*dense_row, false, {});
    auto sparse_row = tatami::convert_to_compressed_sparse<double, int>(*dense_row, true, {});
    auto sparse_col = tatami::convert_to_compressed_sparse<double, int>(*sparse_row, false, {});

    auto rhs = tatami_test::simulate_vector<double>(NC, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 42 + nthreads;
        return opt;
    }());

    tatami_mult::MultiplyWithSingleVectorOptions opt;
    opt.num_threads = nthreads;

    std::vector<double> drout(NR), dcout(NR), srout(NR), scout(NR);
    tatami_mult::multiply_with_single_vector(*dense_row, rhs.data(), drout.data(), opt);
    tatami_mult::multiply_with_single_vector(*dense_col, rhs.data(), dcout.data(), opt);
    tatami_mult::multiply_with_single_vector(*sparse_row, rhs.data(), srout.data(), opt);
    tatami_mult::multiply_with_single_vector(*sparse_col, rhs.data(), scout.data(), opt);

    for (int r = 0; r < NR; ++r) {
        EXPECT_FLOAT_EQ(drout[r], dcout[r]);
        EXPECT_FLOAT_EQ(drout[r], srout[r]);
        EXPECT_FLOAT_EQ(drout[r], scout[r]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    SingleVector,
    SingleVectorDispatchTest,
    ::testing::Combine(
        ::testing::Values(87, 48),
        ::testing::Values(23, 99),
        ::testing::Values(1, 3)
    )
);
