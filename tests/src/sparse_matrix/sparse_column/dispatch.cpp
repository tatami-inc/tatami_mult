#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "tatami_test/tatami_test.hpp"

#include "tatami_mult/sparse_matrix/sparse_column/dispatch.hpp"

#include "../../utils.h"

class SparseMatrixSparseColumnTest : public ::testing::TestWithParam<std::tuple<int, int, int, int> > {};

TEST_P(SparseMatrixSparseColumnTest, Basic) {
    const auto params = GetParam();
    const int NR = std::get<0>(params);
    const int NC = std::get<1>(params);
    const int NRHS = std::get<2>(params);
    const auto nthreads = std::get<3>(params);

    auto dump = tatami_test::simulate_vector<double>(NR * NC, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.density = 0.23;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 693 + NR + NC + NRHS + nthreads;
        return opt;
    }());
    auto sparse_row = tatami::convert_to_compressed_sparse<double, int>(tatami::DenseRowMatrix<double, int>(NR, NC, dump), true, {});
    auto sparse_col = tatami::convert_to_compressed_sparse<double, int>(*sparse_row, false, {});

    auto rhs = tatami_test::simulate_vector<double>(NC * NRHS, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.density = 0.25;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 142 + NR + NC + NRHS + nthreads;
        return opt;
    }());
    auto right_col = tatami::convert_to_compressed_sparse<double, int>(tatami::DenseColumnMatrix<double, int>(NC, NRHS, rhs), false, {});
    auto right_row = tatami::convert_to_compressed_sparse<double, int>(*right_col, true, {});

    tatami_mult::MultiplySparseColumnWithSparseMatrixOptions opt;
    tatami_mult::set_num_threads(opt, nthreads);

    // Setting an initial value for the output vectors, to check that dirty outputs are properly zeroed.
    const auto output_size = NR * NRHS;
    std::vector<double> sr_rr_ro(output_size, 4.5), sr_rr_co(output_size, 5.5),
        sr_rc_ro(output_size, 2.5), sr_rc_co(output_size, 3.5), 
        sc_rr_ro(output_size, 6.5), sc_rr_co(output_size, 7.5),
        sc_rc_ro(output_size, 8.5), sc_rc_co(output_size, 9.5);

    tatami_mult::multiply_sparse_column_with_sparse_matrix(*sparse_col, *right_row, sc_rr_ro.data(), true, opt);
    tatami_mult::multiply_sparse_column_with_sparse_matrix(*sparse_col, *right_row, sc_rr_co.data(), false, opt);
    tatami_mult::multiply_sparse_column_with_sparse_matrix(*sparse_col, *right_col, sc_rc_ro.data(), true, opt);
    tatami_mult::multiply_sparse_column_with_sparse_matrix(*sparse_col, *right_col, sc_rc_co.data(), false, opt);

    // Checking that it still works for row-major LHS.
    tatami_mult::multiply_sparse_column_with_sparse_matrix(*sparse_row, *right_col, sr_rc_ro.data(), true, opt);
    tatami_mult::multiply_sparse_column_with_sparse_matrix(*sparse_row, *right_col, sr_rc_co.data(), false, opt);
    tatami_mult::multiply_sparse_column_with_sparse_matrix(*sparse_row, *right_row, sr_rr_ro.data(), true, opt);
    tatami_mult::multiply_sparse_column_with_sparse_matrix(*sparse_row, *right_row, sr_rr_co.data(), false, opt);

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
    SparseMatrix,
    SparseMatrixSparseColumnTest,
    ::testing::Combine(
        ::testing::Values(98, 36), // number of rows.
        ::testing::Values(35, 104), // number of columns.
        ::testing::Values(11, 46),  // number of RHS vectors.
        ::testing::Values(1, 3)     // number of threads.
    )
);

/******************************/

class SparseMatrixSparseColumnEmptyTest : public ::testing::TestWithParam<std::tuple<int, int, int, int, int> > {};

TEST_P(SparseMatrixSparseColumnEmptyTest, Empty) {
    const auto params = GetParam();
    const int NR = std::get<0>(params);
    const int NC = std::get<1>(params);
    const int NRHS = std::get<2>(params);
    const auto stride = std::get<3>(params);
    const auto nthreads = std::get<4>(params);

    auto dump = simulate_strided_sparse_matrix(NR, NC, stride, /* seed = */ 453 + NR + NC + NRHS + stride + nthreads);
    auto sparse_row = tatami::convert_to_compressed_sparse<double, int>(tatami::DenseRowMatrix<double, int>(NR, NC, dump), true, {});
    auto sparse_col = tatami::convert_to_compressed_sparse<double, int>(*sparse_row, false, {});

    // Empty RHS, check that we can handle rows/columns with no non-zeros.
    auto rhs = simulate_strided_sparse_matrix(NRHS, NC, stride, /* seed = */ 231 + NR + NC + NRHS + stride + nthreads);
    auto right_col = tatami::convert_to_compressed_sparse<double, int>(tatami::DenseColumnMatrix<double, int>(NC, NRHS, rhs), false, {});
    auto right_row = tatami::convert_to_compressed_sparse<double, int>(*right_col, true, {});

    tatami_mult::MultiplySparseColumnWithSparseMatrixOptions opt;
    tatami_mult::set_num_threads(opt, nthreads);

    // Setting an initial value for the output vectors, to check that dirty outputs are properly zeroed.
    const auto output_size = NR * NRHS;
    std::vector<double> sr_rc_ro(output_size, 2.5), sr_rc_co(output_size, 3.5),
        sr_rr_ro(output_size, 4.5), sr_rr_co(output_size, 5.5),
        sc_rr_ro(output_size, 6.5), sc_rr_co(output_size, 7.5),
        sc_rc_ro(output_size, 8.5), sc_rc_co(output_size, 9.5);

    tatami_mult::multiply_sparse_column_with_sparse_matrix(*sparse_col, *right_row, sc_rr_ro.data(), true, opt);
    tatami_mult::multiply_sparse_column_with_sparse_matrix(*sparse_col, *right_row, sc_rr_co.data(), false, opt);
    tatami_mult::multiply_sparse_column_with_sparse_matrix(*sparse_col, *right_col, sc_rc_ro.data(), true, opt);
    tatami_mult::multiply_sparse_column_with_sparse_matrix(*sparse_col, *right_col, sc_rc_co.data(), false, opt);

    // Checking that it still works for row-major LHS.
    tatami_mult::multiply_sparse_column_with_sparse_matrix(*sparse_row, *right_col, sr_rc_ro.data(), true, opt);
    tatami_mult::multiply_sparse_column_with_sparse_matrix(*sparse_row, *right_col, sr_rc_co.data(), false, opt);
    tatami_mult::multiply_sparse_column_with_sparse_matrix(*sparse_row, *right_row, sr_rr_ro.data(), true, opt);
    tatami_mult::multiply_sparse_column_with_sparse_matrix(*sparse_row, *right_row, sr_rr_co.data(), false, opt);

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
    SparseMatrix,
    SparseMatrixSparseColumnEmptyTest,
    ::testing::Combine(
        ::testing::Values(88, 67), // number of rows.
        ::testing::Values(55, 102), // number of columns.
        ::testing::Values(23, 98),  // number of RHS vectors.
        ::testing::Values(0, 3, 10), // non-empty stride. 
        ::testing::Values(1, 3)  // number of threads.
    )
);

/******************************/

TEST(SparseMatrixSparseColumn, Options) {
    tatami_mult::MultiplySparseColumnWithSparseMatrixOptions opt;
    tatami_mult::set_num_threads(opt, 12);
    EXPECT_EQ(opt.column_to_column.num_threads, 12);
    EXPECT_EQ(opt.column_to_row.num_threads, 12);
    EXPECT_EQ(opt.row_to_column.num_threads, 12);
    EXPECT_EQ(opt.row_to_row.num_threads, 12);
}
