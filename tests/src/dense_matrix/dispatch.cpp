#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "tatami_test/tatami_test.hpp"

#include "tatami_mult/dense_matrix/dispatch.hpp"

class DenseMatrixDispatchTest : public ::testing::TestWithParam<std::tuple<int, int, int, int, int> > {};

TEST_P(DenseMatrixDispatchTest, Basic) {
    const auto params = GetParam();
    const int NR = std::get<0>(params);
    const int NC = std::get<1>(params);
    const int NRHS = std::get<2>(params);
    const auto block_size = std::get<3>(params);
    const auto nthreads = std::get<4>(params);

    auto dump = tatami_test::simulate_vector<double>(NR * NC, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.density = 0.25;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 669 + NR + NC + NRHS + block_size + nthreads;
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
        opt.seed = 442 + NR + NC + NRHS + block_size + nthreads;
        return opt;
    }());
    auto right_col = std::make_unique<tatami::DenseColumnMatrix<double, int> >(NC, NRHS, rhs);

    tatami_mult::MultiplyWithDenseMatrixOptions opt;
//    opt.column_to_column.num_threads = nthreads;
//    opt.column_to_column.block_size = block_size;
//    opt.column_to_row.num_threads = nthreads;
//    opt.column_to_row.block_size = block_size;
//    opt.row_to_column.num_threads = nthreads;
//    opt.row_to_row.num_threads = nthreads;

    const auto output_size = NR * NRHS;
    std::vector<double> dr_ro(output_size), dr_co(output_size),
        dc_ro(output_size), dc_co(output_size),
        sr_ro(output_size), sr_co(output_size),
        sc_ro(output_size), sc_co(output_size);

    tatami_mult::multiply_with_dense_matrix(*dense_row, *right_col, dr_ro.data(), true, opt);
    tatami_mult::multiply_with_dense_matrix(*dense_col, *right_col, dc_ro.data(), true, opt);
    tatami_mult::multiply_with_dense_matrix(*sparse_row, *right_col, sr_ro.data(), true, opt);
    tatami_mult::multiply_with_dense_matrix(*sparse_col, *right_col, sc_ro.data(), true, opt);

    tatami_mult::multiply_with_dense_matrix(*dense_row, *right_col, dr_co.data(), false, opt);
    tatami_mult::multiply_with_dense_matrix(*dense_col, *right_col, dc_co.data(), false, opt);
    tatami_mult::multiply_with_dense_matrix(*sparse_row, *right_col, sr_co.data(), false, opt);
    tatami_mult::multiply_with_dense_matrix(*sparse_col, *right_col, sc_co.data(), false, opt);

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

INSTANTIATE_TEST_SUITE_P(
    DenseMatrix,
    DenseMatrixDispatchTest,
    ::testing::Combine(
        ::testing::Values(100, 33), // number of rows.
        ::testing::Values(59, 148), // number of columns.
        ::testing::Values(12, 74),  // number of RHS vectors.
        ::testing::Values(1, 4, 8), // block size.
        ::testing::Values(1, 3)
    )
);
