#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "tatami_test/tatami_test.hpp"

#include "tatami_mult/dense_matrix/sparse_column/dispatch.hpp"

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
        opt.seed = 69 + NR + NC + NRHS + block_size + nthreads;
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

    tatami_mult::MultiplySparseColumnWithDenseMatrixOptions opt;
    opt.column_to_column.num_threads = nthreads;
    opt.column_to_column.block_size = block_size;
    opt.column_to_row.num_threads = nthreads;
    opt.row_to_column.num_threads = nthreads;
    opt.row_to_column.block_size = block_size;
    opt.row_to_row.num_threads = nthreads;

    const auto output_size = NR * NRHS;
    std::vector<double> sr_rc_ro(output_size), sr_rc_co(output_size),
        sr_rr_ro(output_size), sr_rr_co(output_size),
        sc_rr_ro(output_size), sc_rr_co(output_size),
        sc_rc_ro(output_size), sc_rc_co(output_size);

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
        ::testing::Values(1, 3)
    )
);

/************************************/

class DenseMatrixSparseColumnEmptyTest : public ::testing::TestWithParam<std::tuple<int, int> > {};

TEST_P(DenseMatrixSparseColumnEmptyTest, Basic) {
    const int NR = 72;
    const int NC = 123;
    const int NRHS = 11;

    const auto params = GetParam();
    const auto block_size = std::get<0>(params);
    const auto nthreads = std::get<1>(params);

    // Getting some code coverage for an empty sparse matrix (i.e., all values are zero),
    // where column-to-row has some special code to avoid extra work if an LHS column has no non-zeros.
    std::vector<double> dump(NR * NC);
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

    const auto output_size = NR * NRHS;
    std::vector<double> sr_rc_ro(output_size), sr_rc_co(output_size),
        sr_rr_ro(output_size), sr_rr_co(output_size),
        sc_rr_ro(output_size), sc_rr_co(output_size),
        sc_rc_ro(output_size), sc_rc_co(output_size);

    tatami_mult::MultiplySparseColumnWithDenseMatrixOptions opt;
    opt.column_to_column.num_threads = nthreads;
    opt.column_to_column.block_size = block_size;
    opt.column_to_row.num_threads = nthreads;
    opt.row_to_column.num_threads = nthreads;
    opt.row_to_column.block_size = block_size;
    opt.row_to_row.num_threads = nthreads;

    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_row, *right_col, sr_rc_ro.data(), true, opt);
    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_row, *right_row, sr_rr_ro.data(), true, opt);
    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_row, *right_col, sr_rc_co.data(), false, opt);
    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_row, *right_row, sr_rr_co.data(), false, opt);

    // Checking that it still works for column-major LHS.
    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_col, *right_row, sc_rr_ro.data(), true, opt);
    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_col, *right_col, sc_rc_ro.data(), true, opt);
    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_col, *right_row, sc_rr_co.data(), false, opt);
    tatami_mult::multiply_sparse_column_with_dense_matrix(*sparse_col, *right_col, sc_rc_co.data(), false, opt);

    std::vector<double> zeroed(output_size);
    EXPECT_EQ(sr_rc_ro, zeroed);
    EXPECT_EQ(sr_rc_co, zeroed);
    EXPECT_EQ(sr_rr_ro, zeroed);
    EXPECT_EQ(sr_rr_co, zeroed);

    EXPECT_EQ(sc_rc_ro, zeroed);
    EXPECT_EQ(sc_rc_co, zeroed);
    EXPECT_EQ(sc_rr_ro, zeroed);
    EXPECT_EQ(sc_rr_co, zeroed);
}

INSTANTIATE_TEST_SUITE_P(
    DenseMatrix,
    DenseMatrixSparseColumnEmptyTest,
    ::testing::Combine(
        ::testing::Values(1, 4, 8), // block size.
        ::testing::Values(1, 3)
    )
);

