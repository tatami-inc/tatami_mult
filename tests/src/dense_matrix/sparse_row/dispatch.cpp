#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "tatami_test/tatami_test.hpp"

#include "tatami_mult/dense_matrix/sparse_row/dispatch.hpp"

class DenseMatrixSparseRowTest : public ::testing::TestWithParam<std::tuple<int, int, int, int, int> > {};

TEST_P(DenseMatrixSparseRowTest, Vector) {
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
        opt.seed = 69 + NR + NC + block_size + nthreads;
        return opt;
    }());
    auto sparse_row = tatami::convert_to_compressed_sparse<double, int>(tatami::DenseRowMatrix<double, int>(NR, NC, dump), true, {});
    auto sparse_col = tatami::convert_to_compressed_sparse<double, int>(*sparse_row, false, {});

    auto rhs = tatami_test::simulate_vector<double>(NC * NRHS, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 42 + NR + NC + block_size + nthreads;
        return opt;
    }());
    auto right_col = std::make_unique<tatami::DenseColumnMatrix<double, int> >(NC, NRHS, rhs);
    auto right_row = tatami::convert_to_dense<double, int>(*right_col, true, {});

    tatami_mult::MultiplySparseRowWithDenseMatrixOptions opt;
    opt.column_to_column.num_threads = nthreads;
    opt.column_to_column.block_size = block_size;
    opt.column_to_row.num_threads = nthreads;
    opt.column_to_row.block_size = block_size;
    opt.row_to_column.num_threads = nthreads;
    opt.row_to_row.num_threads = nthreads;

    const auto output_size = NR * NRHS;
    std::vector<double> dr_rc_ro1(output_size), dr_rc_ro4(output_size),
        dr_rc_co1(output_size), dr_rc_co4(output_size),
        dr_rr_ro(output_size), dr_rr_co(output_size),
        dc_rr_ro(output_size), dc_rr_co(output_size),
        dc_rc_ro(output_size), dc_rc_co(output_size);

    // Checking different choices of accumulators.
    tatami_mult::multiply_sparse_row_with_dense_matrix<1>(*sparse_row, *right_col, dr_rc_ro1.data(), true, opt);
    tatami_mult::multiply_sparse_row_with_dense_matrix<4>(*sparse_row, *right_col, dr_rc_ro4.data(), true, opt);
    tatami_mult::multiply_sparse_row_with_dense_matrix<1>(*sparse_row, *right_col, dr_rc_co1.data(), false, opt);
    tatami_mult::multiply_sparse_row_with_dense_matrix<4>(*sparse_row, *right_col, dr_rc_co4.data(), false, opt);

    tatami_mult::multiply_sparse_row_with_dense_matrix(*sparse_row, *right_row, dr_rr_ro.data(), true, opt);
    tatami_mult::multiply_sparse_row_with_dense_matrix(*sparse_row, *right_row, dr_rr_co.data(), false, opt);

    // Checking that it still works for column-major LHS.
    tatami_mult::multiply_sparse_row_with_dense_matrix(*sparse_col, *right_row, dc_rr_ro.data(), true, opt);
    tatami_mult::multiply_sparse_row_with_dense_matrix(*sparse_col, *right_col, dc_rc_ro.data(), true, opt);
    tatami_mult::multiply_sparse_row_with_dense_matrix(*sparse_col, *right_row, dc_rr_co.data(), false, opt);
    tatami_mult::multiply_sparse_row_with_dense_matrix(*sparse_col, *right_col, dc_rc_co.data(), false, opt);

    for (int h = 0; h < NRHS; ++h) {
        const auto rptr = rhs.data() + h * NC;
        for (int r = 0; r < NR; ++r) {
            const auto ref = std::inner_product(rptr, rptr + NC, dump.begin() + r * NC, 0.0);

            const auto rm_idx = r * NRHS + h;
            EXPECT_FLOAT_EQ(ref, dr_rc_ro1[rm_idx]);
            EXPECT_FLOAT_EQ(ref, dr_rc_ro4[rm_idx]);
            EXPECT_FLOAT_EQ(ref, dr_rr_ro[rm_idx]);
            EXPECT_FLOAT_EQ(ref, dc_rc_ro[rm_idx]);
            EXPECT_FLOAT_EQ(ref, dc_rr_ro[rm_idx]);

            const auto cm_idx = h * NR + r;
            EXPECT_FLOAT_EQ(ref, dr_rc_co1[cm_idx]);
            EXPECT_FLOAT_EQ(ref, dr_rc_co4[cm_idx]);
            EXPECT_FLOAT_EQ(ref, dr_rr_co[cm_idx]);
            EXPECT_FLOAT_EQ(ref, dc_rc_co[cm_idx]);
            EXPECT_FLOAT_EQ(ref, dc_rr_co[cm_idx]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    DenseMatrix,
    DenseMatrixSparseRowTest,
    ::testing::Combine(
        ::testing::Values(100, 33), // number of rows.
        ::testing::Values(59, 148), // number of columns.
        ::testing::Values(12, 74),  // number of RHS vectors.
        ::testing::Values(1, 4, 8), // block size.
        ::testing::Values(1, 3)
    )
);
