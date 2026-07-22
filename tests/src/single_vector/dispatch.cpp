#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "tatami_test/tatami_test.hpp"

#include "tatami_mult/single_vector/dispatch.hpp"

TEST(SingleVectorDispatch, Vector) {
    const int NR = 87;
    const int NC = 99;

    auto dump = tatami_test::simulate_vector<double>(NR * NC, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.density = 0.19;
        opt.seed = 69;
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
        opt.seed = 42;
        return opt;
    }());

    // Setting an arbitrary initial value for all output vectors, to check that they are properly zeroed.
    std::vector<double> drout(NR), dcout(NR), srout(NR), scout(NR);

    tatami_mult::multiply_with_single_vector(*dense_row, rhs.data(), drout.data(), {});
    tatami_mult::multiply_with_single_vector(*dense_col, rhs.data(), dcout.data(), {});
    tatami_mult::multiply_with_single_vector(*sparse_row, rhs.data(), srout.data(), {});
    tatami_mult::multiply_with_single_vector(*sparse_col, rhs.data(), scout.data(), {});

    for (int r = 0; r < NR; ++r) {
        EXPECT_FLOAT_EQ(drout[r], dcout[r]);
        EXPECT_FLOAT_EQ(drout[r], srout[r]);
        EXPECT_FLOAT_EQ(drout[r], scout[r]);
    }

    // Checking the transpose.
    std::vector<double> trout(NR);
    auto transposed = std::make_unique<tatami::DenseColumnMatrix<double, int> >(NC, NR, dump);
    tatami_mult::multiply_with_single_vector(rhs.data(), *transposed, trout.data(), {});
    EXPECT_EQ(trout, drout);
}

TEST(SingleVectorDispatch, Options) {
    tatami_mult::MultiplyWithSingleVectorOptions opt;
    tatami_mult::set_num_threads(opt, 13);
    EXPECT_EQ(opt.dense_row.num_threads, 13);
    EXPECT_EQ(opt.dense_column.num_threads, 13);
    EXPECT_EQ(opt.sparse_row.num_threads, 13);
    EXPECT_EQ(opt.sparse_column.num_threads, 13);
}
