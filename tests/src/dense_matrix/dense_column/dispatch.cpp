#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "tatami_test/tatami_test.hpp"

#include "tatami_mult/dense_matrix/dense_column/dispatch.hpp"

class DenseMatrixDenseColumnTest : public ::testing::TestWithParam<std::tuple<int, int, int, std::pair<int, int>, int> > {};

TEST_P(DenseMatrixDenseColumnTest, Basic) {
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
        opt.seed = 69 + NR + NC + NRHS + blocks.first + blocks.second + nthreads;
        return opt;
    }());
    auto dense_row = std::make_unique<tatami::DenseRowMatrix<double, int> >(NR, NC, dump);
    auto dense_col = tatami::convert_to_dense<double, int>(*dense_row, false, {});

    auto rhs = tatami_test::simulate_vector<double>(NC * NRHS, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 42 + NR + NC + NRHS + blocks.first + blocks.second + nthreads;
        return opt;
    }());
    auto right_col = std::make_unique<tatami::DenseColumnMatrix<double, int> >(NC, NRHS, rhs);

    std::vector<double> t_rhs(rhs.size());
    tatami::transpose(rhs.data(), NRHS, NC, t_rhs.data());
    auto right_row = std::make_unique<tatami::DenseRowMatrix<double, int> >(NC, NRHS, t_rhs);

    tatami_mult::MultiplyDenseColumnWithDenseMatrixOptions opt;
    tatami_mult::set_num_threads(opt, nthreads);
    tatami_mult::set_dense_primary_block_size(opt, blocks.first);
    tatami_mult::set_dense_secondary_block_size(opt, blocks.second);

    // Setting an initial value for the output vectors, to check that dirty outputs are properly zeroed.
    const auto output_size = NR * NRHS;
    std::vector<double> dc_rr_ro(output_size, 1.2), dc_rr_co(output_size, 1.2),
        dc_rc_ro(output_size, 2.3), dc_rc_co(output_size, 2.3),
        dr_rr_ro(output_size, 3.4), dr_rr_co(output_size, 3.4),
        dr_rc_ro(output_size, 4.5), dr_rc_co(output_size, 4.5);

    tatami_mult::multiply_dense_column_with_dense_matrix(*dense_col, *right_col, dc_rc_ro.data(), true, opt);
    tatami_mult::multiply_dense_column_with_dense_matrix(*dense_col, *right_col, dc_rc_co.data(), false, opt);
    tatami_mult::multiply_dense_column_with_dense_matrix(*dense_col, *right_row, dc_rr_ro.data(), true, opt);
    tatami_mult::multiply_dense_column_with_dense_matrix(*dense_col, *right_row, dc_rr_co.data(), false, opt);

    // Checking that it still works for row-major LHS.
    tatami_mult::multiply_dense_column_with_dense_matrix(*dense_row, *right_col, dr_rc_ro.data(), true, opt);
    tatami_mult::multiply_dense_column_with_dense_matrix(*dense_row, *right_col, dr_rc_co.data(), false, opt);
    tatami_mult::multiply_dense_column_with_dense_matrix(*dense_row, *right_row, dr_rr_ro.data(), true, opt);
    tatami_mult::multiply_dense_column_with_dense_matrix(*dense_row, *right_row, dr_rr_co.data(), false, opt);

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

    // Check the functor overloads that aren't directly called.
    auto get_right = [&](std::size_t rr) -> const double* {
        return t_rhs.data() + rr * NRHS;
    };
    std::vector<double> fun_rr_ro(output_size, 1.2), fun_rr_co(output_size, 2.2);
    tatami_mult::multiply_dense_column_with_dense_row_matrix_to_column_output(*dense_col, NRHS, get_right, fun_rr_co.data(), opt.row_to_column);
    EXPECT_EQ(dc_rr_co, fun_rr_co);
    tatami_mult::multiply_dense_column_with_dense_row_matrix_to_column_output(*dense_row, NRHS, get_right, fun_rr_co.data(), opt.row_to_column);
    EXPECT_EQ(dr_rr_co, fun_rr_co);
    tatami_mult::multiply_dense_column_with_dense_row_matrix_to_row_output(*dense_col, NRHS, get_right, fun_rr_ro.data(), opt.row_to_row);
    EXPECT_EQ(dc_rr_ro, fun_rr_ro);
    tatami_mult::multiply_dense_column_with_dense_row_matrix_to_row_output(*dense_row, NRHS, get_right, fun_rr_ro.data(), opt.row_to_row);
    EXPECT_EQ(dr_rr_ro, fun_rr_ro);
}

INSTANTIATE_TEST_SUITE_P(
    DenseMatrix,
    DenseMatrixDenseColumnTest,
    ::testing::Combine(
        ::testing::Values(120, 15), // number of rows.
        ::testing::Values(13, 152), // number of columns.
        ::testing::Values(9, 84),   // number of RHS vectors.
        ::testing::Values(          // block size.
            std::make_pair(1, 0),
            std::make_pair(4, 16),
            std::make_pair(8, 8)
        ),
        ::testing::Values(1, 3)
    )
);

/************************************/

TEST(DenseMatrixDenseColumn, Options) {
    tatami_mult::MultiplyDenseColumnWithDenseMatrixOptions opt;
    tatami_mult::set_num_threads(opt, 12);
    EXPECT_EQ(opt.column_to_column.num_threads, 12);
    EXPECT_EQ(opt.column_to_row.num_threads, 12);
    EXPECT_EQ(opt.row_to_column.num_threads, 12);
    EXPECT_EQ(opt.row_to_row.num_threads, 12);

    tatami_mult::set_dense_primary_block_size(opt, 23);
    EXPECT_EQ(opt.column_to_column.primary_block_size, 23);
    EXPECT_EQ(opt.column_to_row.primary_block_size, 23);
    EXPECT_EQ(opt.row_to_column.primary_block_size, 23);
    EXPECT_EQ(opt.row_to_row.primary_block_size, 23);

    tatami_mult::set_dense_secondary_block_size(opt, 34);
    EXPECT_EQ(opt.column_to_column.secondary_block_size, 34);
    EXPECT_EQ(opt.column_to_row.secondary_block_size, 34);
    EXPECT_EQ(opt.row_to_column.secondary_block_size, 34);
    EXPECT_EQ(opt.row_to_row.secondary_block_size, 34);
}

/************************************/

TEST(DenseMatrixDenseColumn, UnsignedFunctor) {
    // Check that the functor overloads are compiled without warnings for different combinations of integer signedness.
    // (Note that compilation of the regular overloads are handled in the top-level tatami_mult.cpp, so we don't need to check those.)
    const int NR = 47;
    const int NC = 23;
    const int NRHS = 98;

    auto dump = tatami_test::simulate_vector<double>(NR * NC, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 11 + NR + NC + NRHS;
        return opt;
    }());
    tatami::DenseColumnMatrix<double, int> lhs(NR, NC, dump);
    tatami::DenseColumnMatrix<double, unsigned> lhs_u(NR, NC, std::move(dump));

    auto rdump = tatami_test::simulate_vector<double>(NC * NRHS, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 13 + NR + NC + NRHS;
        return opt;
    }());
    auto get_right = [&](std::size_t rr) -> const double* {
        return rdump.data() + rr * NRHS;
    };

    const auto output_size = NR * NRHS;
    std::vector<double> ref(output_size), out_us(output_size), out_su(output_size), out_uu(output_size);

    tatami_mult::multiply_dense_column_with_dense_row_matrix_to_column_output(lhs, NRHS, get_right, ref.data(), {});
    tatami_mult::multiply_dense_column_with_dense_row_matrix_to_column_output(lhs, (unsigned)NRHS, get_right, out_su.data(), {});
    EXPECT_EQ(ref, out_su);
    tatami_mult::multiply_dense_column_with_dense_row_matrix_to_column_output(lhs_u, NRHS, get_right, out_us.data(), {});
    EXPECT_EQ(ref, out_us);
    tatami_mult::multiply_dense_column_with_dense_row_matrix_to_column_output(lhs_u, (unsigned)NRHS, get_right, out_uu.data(), {});
    EXPECT_EQ(ref, out_uu);

    tatami_mult::multiply_dense_column_with_dense_row_matrix_to_row_output(lhs, NRHS, get_right, ref.data(), {});
    tatami_mult::multiply_dense_column_with_dense_row_matrix_to_row_output(lhs, (unsigned)NRHS, get_right, out_su.data(), {});
    EXPECT_EQ(ref, out_su);
    tatami_mult::multiply_dense_column_with_dense_row_matrix_to_row_output(lhs_u, NRHS, get_right, out_us.data(), {});
    EXPECT_EQ(ref, out_us);
    tatami_mult::multiply_dense_column_with_dense_row_matrix_to_row_output(lhs_u, (unsigned)NRHS, get_right, out_uu.data(), {});
    EXPECT_EQ(ref, out_uu);
}
