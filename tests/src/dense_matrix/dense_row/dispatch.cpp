#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "tatami_test/tatami_test.hpp"

#include "tatami_mult/dense_matrix/dense_row/dispatch.hpp"

class DenseMatrixDenseRowTest : public ::testing::TestWithParam<std::tuple<int, int, int, std::pair<int, int>, int> > {};

TEST_P(DenseMatrixDenseRowTest, Vector) {
    const auto params = GetParam();
    const int NR = std::get<0>(params);
    const int NC = std::get<1>(params);
    const int NRHS = std::get<2>(params);
    const auto blocks = std::get<3>(params);
    const auto nthreads = std::get<4>(params);

    auto dump = tatami_test::simulate_vector<double>(NR * NC, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 69 + NR + NC + blocks.first + blocks.second + nthreads;
        return opt;
    }());
    auto dense_row = std::make_unique<tatami::DenseRowMatrix<double, int> >(NR, NC, dump);
    auto dense_col = tatami::convert_to_dense<double, int>(*dense_row, false, {});

    auto rhs = tatami_test::simulate_vector<double>(NC * NRHS, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 42 + NR + NC + blocks.first + blocks.second + nthreads;
        return opt;
    }());
    auto right_col = std::make_unique<tatami::DenseColumnMatrix<double, int> >(NC, NRHS, rhs);
    auto right_row = tatami::convert_to_dense<double, int>(*right_col, true, {});

    tatami_mult::MultiplyDenseRowWithDenseMatrixOptions opt;
    opt.column_to_column.num_threads = nthreads;
    opt.column_to_column.primary_block_size = blocks.first;
    opt.column_to_column.secondary_block_size = blocks.second;
    opt.column_to_row.num_threads = nthreads;
    opt.column_to_row.primary_block_size = blocks.first;
    opt.column_to_row.secondary_block_size = blocks.second;
    opt.row_to_column.num_threads = nthreads;
    opt.row_to_column.primary_block_size = blocks.first;
    opt.row_to_column.secondary_block_size = blocks.second;
    opt.row_to_row.num_threads = nthreads;
    opt.row_to_row.primary_block_size = blocks.first;
    opt.row_to_row.secondary_block_size = blocks.second;

    const auto output_size = NR * NRHS;
    std::vector<double> dr_rc_ro1(output_size), dr_rc_ro4(output_size),
        dr_rc_co1(output_size), dr_rc_co4(output_size),
        dr_rr_ro(output_size), dr_rr_co(output_size),
        dc_rr_ro(output_size), dc_rr_co(output_size),
        dc_rc_ro(output_size), dc_rc_co(output_size);

    // Checking different choices of accumulators.
    tatami_mult::multiply_dense_row_with_dense_matrix<1>(*dense_row, *right_col, dr_rc_ro1.data(), true, opt);
    tatami_mult::multiply_dense_row_with_dense_matrix<4>(*dense_row, *right_col, dr_rc_ro4.data(), true, opt);
    tatami_mult::multiply_dense_row_with_dense_matrix<1>(*dense_row, *right_col, dr_rc_co1.data(), false, opt);
    tatami_mult::multiply_dense_row_with_dense_matrix<4>(*dense_row, *right_col, dr_rc_co4.data(), false, opt);

    tatami_mult::multiply_dense_row_with_dense_matrix(*dense_row, *right_row, dr_rr_ro.data(), true, opt);
    tatami_mult::multiply_dense_row_with_dense_matrix(*dense_row, *right_row, dr_rr_co.data(), false, opt);

    // Checking that it still works for column-major LHS.
    tatami_mult::multiply_dense_row_with_dense_matrix(*dense_col, *right_row, dc_rr_ro.data(), true, opt);
    tatami_mult::multiply_dense_row_with_dense_matrix(*dense_col, *right_col, dc_rc_ro.data(), true, opt);
    tatami_mult::multiply_dense_row_with_dense_matrix(*dense_col, *right_row, dc_rr_co.data(), false, opt);
    tatami_mult::multiply_dense_row_with_dense_matrix(*dense_col, *right_col, dc_rc_co.data(), false, opt);

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
    DenseMatrixDenseRowTest,
    ::testing::Combine(
        ::testing::Values(100, 13), // number of rows.
        ::testing::Values(14, 148), // number of columns.
        ::testing::Values(10, 74),  // number of RHS vectors.
        ::testing::Values(          // block size.
            std::make_pair(1, 0),
            std::make_pair(4, 16),
            std::make_pair(8, 8)
        ),
        ::testing::Values(1, 3)
    )
);
