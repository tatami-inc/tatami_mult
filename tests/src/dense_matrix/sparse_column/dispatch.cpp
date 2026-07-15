#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "tatami_test/tatami_test.hpp"

#include "tatami_mult/dense_matrix/sparse_column/dispatch.hpp"

#include "../../utils.h"

class DenseMatrixSparseColumnTest : public ::testing::TestWithParam<std::tuple<int, int, int, int, int> > {};

TEST_P(DenseMatrixSparseColumnTest, Basic) {
    const auto params = GetParam();
    const int NR = std::get<0>(params);
    const int NC = std::get<1>(params);
    const int NRHS = std::get<2>(params);
    const auto block_size = std::get<3>(params);
    const auto nthreads = std::get<4>(params);

    auto dump = tatami_test::simulate_vector<double>(NR * NC, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.density = 0.1;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 369 + NR + NC + NRHS + block_size + nthreads;
        return opt;
    }());
    auto sparse_row = tatami::convert_to_compressed_sparse<double, int>(tatami::DenseRowMatrix<double, int>(NR, NC, dump), true, {});
    auto sparse_col = tatami::convert_to_compressed_sparse<double, int>(*sparse_row, false, {});

    auto rhs = tatami_test::simulate_vector<double>(NC * NRHS, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 420 + NR + NC + NRHS + block_size + nthreads;
        return opt;
    }());
    auto right_col = std::make_unique<tatami::DenseColumnMatrix<double, int> >(NC, NRHS, rhs);
    auto right_row = tatami::convert_to_dense<double, int>(*right_col, true, {});

    tatami_mult::MultiplySparseColumnWithDenseMatrixOptions opt;
    opt.column_to_column.num_threads = nthreads;
    opt.column_to_column.block_size = block_size;
    opt.column_to_row.num_threads = nthreads;
    opt.row_to_column.num_threads = nthreads;
    opt.row_to_column.block_size = block_size;
    opt.row_to_row.num_threads = nthreads;

    // Setting an initial value for the output vectors, to check that dirty outputs are properly zeroed.
    const auto output_size = NR * NRHS;
    std::vector<double> sr_rc_ro(output_size, 3.2), sr_rc_co(output_size, 2.1),
        sr_rr_ro(output_size, 1.0), sr_rr_co(output_size, 0.9),
        sc_rr_ro(output_size, 9.8), sc_rr_co(output_size, 8.7),
        sc_rc_ro(output_size, 7.6), sc_rc_co(output_size, 6.5);

    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_row, *right_col, sr_rc_ro.data(), true, opt);
    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_row, *right_row, sr_rr_ro.data(), true, opt);
    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_row, *right_col, sr_rc_co.data(), false, opt);
    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_row, *right_row, sr_rr_co.data(), false, opt);

    // Checking that it still works for column-major LHS.
    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_col, *right_row, sc_rr_ro.data(), true, opt);
    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_col, *right_col, sc_rc_ro.data(), true, opt);
    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_col, *right_row, sc_rr_co.data(), false, opt);
    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_col, *right_col, sc_rc_co.data(), false, opt);

    for (int h = 0; h < NRHS; ++h) {
        const auto rptr = rhs.data() + h * NC;
        for (int r = 0; r < NR; ++r) {
            const auto ref = std::inner_product(rptr, rptr + NC, dump.begin() + r * NC, 0.0);

            const auto rm_idx = r * NRHS + h;
            EXPECT_FLOAT_EQ(ref, sr_rc_ro[rm_idx]);
            EXPECT_FLOAT_EQ(ref, sr_rr_ro[rm_idx]);
            EXPECT_FLOAT_EQ(ref, sc_rc_ro[rm_idx]);
            EXPECT_FLOAT_EQ(ref, sc_rr_ro[rm_idx]);

            const auto cm_idx = h * NR + r;
            EXPECT_FLOAT_EQ(ref, sr_rc_co[cm_idx]);
            EXPECT_FLOAT_EQ(ref, sr_rr_co[cm_idx]);
            EXPECT_FLOAT_EQ(ref, sc_rc_co[cm_idx]);
            EXPECT_FLOAT_EQ(ref, sc_rr_co[cm_idx]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    DenseMatrix,
    DenseMatrixSparseColumnTest,
    ::testing::Combine(
        ::testing::Values(100, 33), // number of rows.
        ::testing::Values(59, 148), // number of columns.
        ::testing::Values(12, 74),  // number of RHS vectors.
        ::testing::Values(1, 4, 8), // block size.
        ::testing::Values(1, 3) // number of threads.
    )
);

/************************************/

class DenseMatrixSparseColumnEmptyTest : public ::testing::TestWithParam<std::tuple<int, int, int, int, int, int> > {};

TEST_P(DenseMatrixSparseColumnEmptyTest, Basic) {
    const auto params = GetParam();
    const int NR = std::get<0>(params);
    const int NC = std::get<1>(params);
    const int NRHS = std::get<2>(params);
    const auto stride = std::get<3>(params);
    const auto block_size = std::get<4>(params);
    const auto nthreads = std::get<5>(params);

    auto dump = simulate_strided_sparse_matrix(NR, NC, stride, /* seed = */ 923 + NR + NC + NRHS + block_size + nthreads);
    auto sparse_row = tatami::convert_to_compressed_sparse<double, int>(tatami::DenseRowMatrix<double, int>(NR, NC, dump), true, {});
    auto sparse_col = tatami::convert_to_compressed_sparse<double, int>(*sparse_row, false, {});

    auto rhs = tatami_test::simulate_vector<double>(NC * NRHS, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 42 + block_size + nthreads;
        return opt;
    }());
    auto right_col = std::make_unique<tatami::DenseColumnMatrix<double, int> >(NC, NRHS, rhs);
    auto right_row = tatami::convert_to_dense<double, int>(*right_col, true, {});

    // Setting an initial value for the output vectors, to check that dirty outputs are properly zeroed.
    const auto output_size = NR * NRHS;
    std::vector<double> sr_rc_ro(output_size, 5.5), sr_rc_co(output_size, 4.4),
        sr_rr_ro(output_size, 3.3), sr_rr_co(output_size, 2.2),
        sc_rr_ro(output_size, 1.1), sc_rr_co(output_size, 9.9),
        sc_rc_ro(output_size, 8.8), sc_rc_co(output_size, 7.7);

    tatami_mult::MultiplySparseColumnWithDenseMatrixOptions opt;
    tatami_mult::set_num_threads(opt, nthreads);
    tatami_mult::set_sparse_block_size(opt, block_size);

    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_row, *right_col, sr_rc_ro.data(), true, opt);
    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_row, *right_row, sr_rr_ro.data(), true, opt);
    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_row, *right_col, sr_rc_co.data(), false, opt);
    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_row, *right_row, sr_rr_co.data(), false, opt);

    // Checking that it still works for column-major LHS.
    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_col, *right_row, sc_rr_ro.data(), true, opt);
    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_col, *right_col, sc_rc_ro.data(), true, opt);
    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_col, *right_row, sc_rr_co.data(), false, opt);
    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_col, *right_col, sc_rc_co.data(), false, opt);

    for (int h = 0; h < NRHS; ++h) {
        const auto rptr = rhs.data() + h * NC;
        for (int r = 0; r < NR; ++r) {
            const auto ref = std::inner_product(rptr, rptr + NC, dump.begin() + r * NC, 0.0);

            const auto rm_idx = r * NRHS + h;
            EXPECT_FLOAT_EQ(ref, sr_rc_ro[rm_idx]);
            EXPECT_FLOAT_EQ(ref, sr_rr_ro[rm_idx]);
            EXPECT_FLOAT_EQ(ref, sc_rc_ro[rm_idx]);
            EXPECT_FLOAT_EQ(ref, sc_rr_ro[rm_idx]);

            const auto cm_idx = h * NR + r;
            EXPECT_FLOAT_EQ(ref, sr_rc_co[cm_idx]);
            EXPECT_FLOAT_EQ(ref, sr_rr_co[cm_idx]);
            EXPECT_FLOAT_EQ(ref, sc_rc_co[cm_idx]);
            EXPECT_FLOAT_EQ(ref, sc_rr_co[cm_idx]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    DenseMatrix,
    DenseMatrixSparseColumnEmptyTest,
    ::testing::Combine(
        ::testing::Values(96, 35), // number of rows.
        ::testing::Values(56, 133), // number of columns.
        ::testing::Values(21, 80),  // number of RHS vectors.
        ::testing::Values(0, 3, 10), // non-empty stride.
        ::testing::Values(1, 4), // block size.
        ::testing::Values(1, 3) // number of threads.
    )
);

/************************************/

TEST(DenseMatrixSparseColumn, Options) {
    tatami_mult::MultiplySparseColumnWithDenseMatrixOptions opt;
    tatami_mult::set_num_threads(opt, 12);
    EXPECT_EQ(opt.column_to_column.num_threads, 12);
    EXPECT_EQ(opt.column_to_row.num_threads, 12);
    EXPECT_EQ(opt.row_to_column.num_threads, 12);
    EXPECT_EQ(opt.row_to_row.num_threads, 12);

    tatami_mult::set_sparse_block_size(opt, 42);
    EXPECT_EQ(opt.column_to_column.block_size, 42);
    EXPECT_EQ(opt.row_to_column.block_size, 42);
}
