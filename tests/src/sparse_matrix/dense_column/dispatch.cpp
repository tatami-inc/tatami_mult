#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "tatami_test/tatami_test.hpp"

#include "tatami_mult/sparse_matrix/dense_column/dispatch.hpp"

#include "../../utils.h"

class SparseMatrixDenseColumnTest : public ::testing::TestWithParam<std::tuple<int, int, int, int, int> > {};

TEST_P(SparseMatrixDenseColumnTest, Basic) {
    const auto params = GetParam();
    const int NR = std::get<0>(params);
    const int NC = std::get<1>(params);
    const int NRHS = std::get<2>(params);
    const auto block_size = std::get<3>(params);
    const auto nthreads = std::get<4>(params);

    auto dump = tatami_test::simulate_vector<double>(NR * NC, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 369 + NR + NC + NRHS + block_size + nthreads;
        return opt;
    }());
    auto dense_row = std::make_unique<tatami::DenseRowMatrix<double, int> >(NR, NC, dump);
    auto dense_col = tatami::convert_to_dense<double, int>(*dense_row, true, {});

    auto rhs = tatami_test::simulate_vector<double>(NC * NRHS, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.density = 0.25;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 420 + NR + NC + NRHS + block_size + nthreads;
        return opt;
    }());
    auto right_col = tatami::convert_to_compressed_sparse<double, int>(tatami::DenseColumnMatrix<double, int>(NC, NRHS, rhs), false, {});
    auto right_row = tatami::convert_to_compressed_sparse<double, int>(*right_col, true, {});

    tatami_mult::MultiplyDenseColumnWithSparseMatrixOptions opt;
    tatami_mult::set_num_threads(opt, nthreads);
    tatami_mult::set_sparse_block_size(opt, block_size);

    // Setting an initial value for the output vectors, to check that dirty outputs are properly zeroed.
    const auto output_size = NR * NRHS;
    std::vector<double> dr_rc_ro(output_size, 2.5), dr_rc_co(output_size, 3.5), 
        dr_rr_ro(output_size, 4.5), dr_rr_co(output_size, 5.5),
        dc_rr_ro(output_size, 6.5), dc_rr_co(output_size, 7.5),
        dc_rc_ro(output_size, 8.5), dc_rc_co(output_size, 9.5);

    tatami_mult::multiply_dense_column_with_sparse_matrix(*dense_col, *right_row, dc_rr_ro.data(), true, opt);
    tatami_mult::multiply_dense_column_with_sparse_matrix(*dense_col, *right_row, dc_rr_co.data(), false, opt);
    tatami_mult::multiply_dense_column_with_sparse_matrix(*dense_col, *right_col, dc_rc_ro.data(), true, opt);
    tatami_mult::multiply_dense_column_with_sparse_matrix(*dense_col, *right_col, dc_rc_co.data(), false, opt);

    // Checking that it still works for row-major LHS.
    tatami_mult::multiply_dense_column_with_sparse_matrix(*dense_row, *right_col, dr_rc_ro.data(), true, opt);
    tatami_mult::multiply_dense_column_with_sparse_matrix(*dense_row, *right_col, dr_rc_co.data(), false, opt);
    tatami_mult::multiply_dense_column_with_sparse_matrix(*dense_row, *right_row, dr_rr_ro.data(), true, opt);
    tatami_mult::multiply_dense_column_with_sparse_matrix(*dense_row, *right_row, dr_rr_co.data(), false, opt);

    for (int h = 0; h < NRHS; ++h) {
        const auto rptr = rhs.data() + h * NC;
        for (int r = 0; r < NR; ++r) {
            const auto ref = std::inner_product(rptr, rptr + NC, dump.begin() + r * NC, 0.0);

            const auto rm_idx = r * NRHS + h;
            EXPECT_FLOAT_EQ(ref, dr_rc_ro[rm_idx]);
            EXPECT_FLOAT_EQ(ref, dr_rr_ro[rm_idx]);
            EXPECT_FLOAT_EQ(ref, dc_rc_ro[rm_idx]);
            EXPECT_FLOAT_EQ(ref, dc_rr_ro[rm_idx]);

            const auto cm_idx = h * NR + r;
            EXPECT_FLOAT_EQ(ref, dr_rc_co[cm_idx]);
            EXPECT_FLOAT_EQ(ref, dr_rr_co[cm_idx]);
            EXPECT_FLOAT_EQ(ref, dc_rc_co[cm_idx]);
            EXPECT_FLOAT_EQ(ref, dc_rr_co[cm_idx]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    SparseMatrix,
    SparseMatrixDenseColumnTest,
    ::testing::Combine(
        ::testing::Values(201, 48), // number of rows.
        ::testing::Values(24, 87), // number of columns.
        ::testing::Values(15, 60),  // number of RHS vectors.
        ::testing::Values(1, 4, 8), // block size.
        ::testing::Values(1, 3)     // number of threads.
    )
);

/******************************/

class SparseMatrixDenseColumnEmptyTest : public ::testing::TestWithParam<std::tuple<int, int, int, int, int, int> > {};

TEST_P(SparseMatrixDenseColumnEmptyTest, Empty) {
    const auto params = GetParam();
    const int NR = std::get<0>(params);
    const int NC = std::get<1>(params);
    const int NRHS = std::get<2>(params);
    const auto stride = std::get<3>(params);
    const auto block_size = std::get<4>(params);
    const auto nthreads = std::get<5>(params);

    auto dump = tatami_test::simulate_vector<double>(NR * NC, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 369 + NR + NC + NRHS + block_size + stride + nthreads;
        return opt;
    }());
    auto dense_row = std::make_unique<tatami::DenseRowMatrix<double, int> >(NR, NC, dump);
    auto dense_col = tatami::convert_to_dense<double, int>(*dense_row, true, {});

    // Empty RHS, check that we can handle rows/columns with no non-zeros.
    auto rhs = simulate_strided_sparse_matrix(NRHS, NC, stride, /* seed = */ 231 + NR + NC + NRHS + block_size + stride + nthreads);
    auto right_col = tatami::convert_to_compressed_sparse<double, int>(tatami::DenseColumnMatrix<double, int>(NC, NRHS, rhs), false, {});
    auto right_row = tatami::convert_to_compressed_sparse<double, int>(*right_col, true, {});

    tatami_mult::MultiplyDenseColumnWithSparseMatrixOptions opt;
    tatami_mult::set_num_threads(opt, nthreads);
    tatami_mult::set_sparse_block_size(opt, block_size);

    // Setting an initial value for the output vectors, to check that dirty outputs are properly zeroed.
    const auto output_size = NR * NRHS;
    std::vector<double> dr_rc_ro(output_size, 2.5), dr_rc_co(output_size, 3.5), 
        dr_rr_ro(output_size, 4.5), dr_rr_co(output_size, 5.5),
        dc_rr_ro(output_size, 6.5), dc_rr_co(output_size, 7.5),
        dc_rc_ro(output_size, 8.5), dc_rc_co(output_size, 9.5);

    tatami_mult::multiply_dense_column_with_sparse_matrix(*dense_col, *right_row, dc_rr_ro.data(), true, opt);
    tatami_mult::multiply_dense_column_with_sparse_matrix(*dense_col, *right_row, dc_rr_co.data(), false, opt);
    tatami_mult::multiply_dense_column_with_sparse_matrix(*dense_col, *right_col, dc_rc_ro.data(), true, opt);
    tatami_mult::multiply_dense_column_with_sparse_matrix(*dense_col, *right_col, dc_rc_co.data(), false, opt);

    // Checking it works with a row-major LHS.
    tatami_mult::multiply_dense_column_with_sparse_matrix(*dense_row, *right_col, dr_rc_ro.data(), true, opt);
    tatami_mult::multiply_dense_column_with_sparse_matrix(*dense_row, *right_col, dr_rc_co.data(), false, opt);
    tatami_mult::multiply_dense_column_with_sparse_matrix(*dense_row, *right_col, dr_rr_ro.data(), true, opt);
    tatami_mult::multiply_dense_column_with_sparse_matrix(*dense_row, *right_col, dr_rr_co.data(), false, opt);

    for (int h = 0; h < NRHS; ++h) {
        const auto rptr = rhs.data() + h * NC;
        for (int r = 0; r < NR; ++r) {
            const auto ref = std::inner_product(rptr, rptr + NC, dump.begin() + r * NC, 0.0);

            const auto rm_idx = r * NRHS + h;
            EXPECT_FLOAT_EQ(ref, dr_rc_ro[rm_idx]);
            EXPECT_FLOAT_EQ(ref, dr_rr_ro[rm_idx]);
            EXPECT_FLOAT_EQ(ref, dc_rc_ro[rm_idx]);
            EXPECT_FLOAT_EQ(ref, dc_rr_ro[rm_idx]);

            const auto cm_idx = h * NR + r;
            EXPECT_FLOAT_EQ(ref, dr_rc_co[cm_idx]);
            EXPECT_FLOAT_EQ(ref, dr_rr_co[cm_idx]);
            EXPECT_FLOAT_EQ(ref, dc_rc_co[cm_idx]);
            EXPECT_FLOAT_EQ(ref, dc_rr_co[cm_idx]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    SparseMatrix,
    SparseMatrixDenseColumnEmptyTest,
    ::testing::Combine(
        ::testing::Values(201, 48), // number of rows.
        ::testing::Values(24, 87), // number of columns.
        ::testing::Values(15, 60),  // number of RHS vectors.
        ::testing::Values(0, 3, 10), // non-empty stride. 
        ::testing::Values(1, 4), // block size.
        ::testing::Values(1, 3)  // number of threads.
    )
);

/******************************/

TEST(SparseMatrixDenseColumn, Options) {
    tatami_mult::MultiplyDenseColumnWithSparseMatrixOptions opt;
    tatami_mult::set_num_threads(opt, 12);
    EXPECT_EQ(opt.column_to_column.num_threads, 12);
    EXPECT_EQ(opt.column_to_row.num_threads, 12);
    EXPECT_EQ(opt.row_to_column.num_threads, 12);
    EXPECT_EQ(opt.row_to_row.num_threads, 12);

    tatami_mult::set_sparse_block_size(opt, 42);
    EXPECT_EQ(opt.row_to_row.block_size, 42);
    EXPECT_EQ(opt.column_to_row.block_size, 42);
}
