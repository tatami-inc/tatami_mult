#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "tatami_test/tatami_test.hpp"

#include "tatami_mult/dense_matrix/dispatch.hpp"

TEST(DenseMatrixDispatch, Basic) {
    const int NR = 100;
    const int NC = 59;
    const int NRHS = 74;

    auto dump = tatami_test::simulate_vector<double>(NR * NC, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.density = 0.25;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 669 + NR + NC + NRHS; 
        return opt;
    }());
    auto dense_row = std::make_unique<tatami::DenseRowMatrix<double, int> >(NR, NC, dump);
    auto dense_col = tatami::convert_to_dense<double, int>(*dense_row, false, {});
    auto sparse_row = tatami::convert_to_compressed_sparse<double, int>(*dense_row, true, {});
    auto sparse_col = tatami::convert_to_compressed_sparse<double, int>(*dense_row, false, {});

    auto rhs = tatami_test::simulate_vector<double>(NC * NRHS, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 442 + NR + NC + NRHS;
        return opt;
    }());
    auto right_col = std::make_unique<tatami::DenseColumnMatrix<double, int> >(NC, NRHS, rhs);

    // Setting an initial value for the output vectors, to check that dirty outputs are properly zeroed.
    const auto output_size = NR * NRHS;
    std::vector<double> dr_ro(output_size, 7.8), dr_co(output_size, 8.9),
        dc_ro(output_size, 9.0), dc_co(output_size, 0.1),
        sr_ro(output_size, 1.2), sr_co(output_size, 2.3),
        sc_ro(output_size, 3.4), sc_co(output_size, 4.5);

    tatami_mult::multiply_with_dense_matrix(*dense_row, *right_col, dr_ro.data(), true, {});
    tatami_mult::multiply_with_dense_matrix(*dense_col, *right_col, dc_ro.data(), true, {});
    tatami_mult::multiply_with_dense_matrix(*sparse_row, *right_col, sr_ro.data(), true, {});
    tatami_mult::multiply_with_dense_matrix(*sparse_col, *right_col, sc_ro.data(), true, {});

    tatami_mult::multiply_with_dense_matrix(*dense_row, *right_col, dr_co.data(), false, {});
    tatami_mult::multiply_with_dense_matrix(*dense_col, *right_col, dc_co.data(), false, {});
    tatami_mult::multiply_with_dense_matrix(*sparse_row, *right_col, sr_co.data(), false, {});
    tatami_mult::multiply_with_dense_matrix(*sparse_col, *right_col, sc_co.data(), false, {});

    for (int h = 0; h < NRHS; ++h) {
        const auto rptr = rhs.data() + h * NC;
        for (int r = 0; r < NR; ++r) {
            const auto ref = std::inner_product(rptr, rptr + NC, dump.begin() + r * NC, 0.0);

            const auto rm_idx = r * NRHS + h;
            EXPECT_FLOAT_EQ(ref, dr_ro[rm_idx]);
            EXPECT_FLOAT_EQ(ref, dc_ro[rm_idx]);
            EXPECT_FLOAT_EQ(ref, sr_ro[rm_idx]);
            EXPECT_FLOAT_EQ(ref, sc_ro[rm_idx]);

            const auto cm_idx = h * NR + r;
            EXPECT_FLOAT_EQ(ref, dr_co[cm_idx]);
            EXPECT_FLOAT_EQ(ref, dc_co[cm_idx]);
            EXPECT_FLOAT_EQ(ref, sr_co[cm_idx]);
            EXPECT_FLOAT_EQ(ref, sc_co[cm_idx]);
        }
    }
}

TEST(DenseMatrixDispatch, Options) {
    tatami_mult::MultiplyWithDenseMatrixOptions opt;
    tatami_mult::set_num_threads(opt, 13);
    EXPECT_EQ(opt.dense_row.column_to_column.num_threads, 13);
    EXPECT_EQ(opt.dense_row.column_to_row.num_threads, 13);
    EXPECT_EQ(opt.dense_row.row_to_column.num_threads, 13);
    EXPECT_EQ(opt.dense_row.row_to_row.num_threads, 13);
    EXPECT_EQ(opt.dense_column.column_to_column.num_threads, 13);
    EXPECT_EQ(opt.dense_column.column_to_row.num_threads, 13);
    EXPECT_EQ(opt.dense_column.row_to_column.num_threads, 13);
    EXPECT_EQ(opt.dense_column.row_to_row.num_threads, 13);
    EXPECT_EQ(opt.sparse_column.column_to_column.num_threads, 13);
    EXPECT_EQ(opt.sparse_column.column_to_row.num_threads, 13);
    EXPECT_EQ(opt.sparse_column.row_to_column.num_threads, 13);
    EXPECT_EQ(opt.sparse_column.row_to_row.num_threads, 13);
    EXPECT_EQ(opt.sparse_row.column_to_column.num_threads, 13);
    EXPECT_EQ(opt.sparse_row.column_to_row.num_threads, 13);
    EXPECT_EQ(opt.sparse_row.row_to_column.num_threads, 13);
    EXPECT_EQ(opt.sparse_row.row_to_row.num_threads, 13);

    tatami_mult::set_dense_primary_block_size(opt, 42);
    EXPECT_EQ(opt.dense_row.column_to_column.primary_block_size, 42);
    EXPECT_EQ(opt.dense_row.column_to_row.primary_block_size, 42);
    EXPECT_EQ(opt.dense_row.row_to_column.primary_block_size, 42);
    EXPECT_EQ(opt.dense_row.row_to_row.primary_block_size, 42);
    EXPECT_EQ(opt.dense_column.column_to_column.primary_block_size, 42);
    EXPECT_EQ(opt.dense_column.column_to_row.primary_block_size, 42);
    EXPECT_EQ(opt.dense_column.row_to_column.primary_block_size, 42);
    EXPECT_EQ(opt.dense_column.row_to_row.primary_block_size, 42);

    tatami_mult::set_dense_secondary_block_size(opt, 69);
    EXPECT_EQ(opt.dense_row.column_to_column.secondary_block_size, 69);
    EXPECT_EQ(opt.dense_row.column_to_row.secondary_block_size, 69);
    EXPECT_EQ(opt.dense_row.row_to_column.secondary_block_size, 69);
    EXPECT_EQ(opt.dense_row.row_to_row.secondary_block_size, 69);
    EXPECT_EQ(opt.dense_column.column_to_column.secondary_block_size, 69);
    EXPECT_EQ(opt.dense_column.column_to_row.secondary_block_size, 69);
    EXPECT_EQ(opt.dense_column.row_to_column.secondary_block_size, 69);
    EXPECT_EQ(opt.dense_column.row_to_row.secondary_block_size, 69);

    tatami_mult::set_sparse_block_size(opt, 100);
    EXPECT_EQ(opt.sparse_row.column_to_column.block_size, 100);
    EXPECT_EQ(opt.sparse_row.column_to_row.block_size, 100);
    EXPECT_EQ(opt.sparse_column.column_to_column.block_size, 100);
    EXPECT_EQ(opt.sparse_column.row_to_column.block_size, 100);
}
