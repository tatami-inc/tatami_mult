#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "tatami_test/tatami_test.hpp"

#include "tatami_mult/multiple_vectors/dispatch.hpp"

static std::vector<double*> formulate_ptrs(const int NR, const int NRHS, std::vector<std::vector<double> >& output) {
    output.resize(NRHS);
    std::vector<double*> ptrs(NRHS);
    for (int h = 0; h < NRHS; ++h) {
        // Setting an initial value for the output vectors, to check that dirty outputs are properly zeroed.
        output[h].resize(NR, 723 + h);
        ptrs[h] = output[h].data();
    }
    return ptrs;
}

TEST(MultipleVectorsDispatch, Basic) {
    const int NR = 107;
    const int NC = 78;
    const int NRHS = 13;

    auto dump = tatami_test::simulate_vector<double>(NR * NC, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.density = 0.24;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 77 + NR + NC + NRHS; 
        return opt;
    }());
    auto dense_row = std::make_unique<tatami::DenseRowMatrix<double, int> >(NR, NC, dump);
    auto dense_col = tatami::convert_to_dense<double, int>(*dense_row, false, {});
    auto sparse_row = tatami::convert_to_compressed_sparse<double, int>(*dense_row, true, {});
    auto sparse_col = tatami::convert_to_compressed_sparse<double, int>(*sparse_row, false, {});

    auto rhs = tatami_test::simulate_vector<double>(NC * NRHS, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 50 + NR + NC + NRHS;
        return opt;
    }());

    std::vector<double*> rhs_ptrs(NRHS);
    for (int h = 0; h < NRHS; ++h) {
        rhs_ptrs[h] = rhs.data() + h * NC;
    }

    std::vector<std::vector<double> > dr_output, dc_output, sr_output, sc_output;
    tatami_mult::multiply_with_multiple_vectors(*dense_row, rhs_ptrs, formulate_ptrs(NR, NRHS, dr_output), {});
    tatami_mult::multiply_with_multiple_vectors(*dense_col, rhs_ptrs, formulate_ptrs(NR, NRHS, dc_output), {});
    tatami_mult::multiply_with_multiple_vectors(*sparse_row, rhs_ptrs, formulate_ptrs(NR, NRHS, sr_output), {});
    tatami_mult::multiply_with_multiple_vectors(*sparse_col, rhs_ptrs, formulate_ptrs(NR, NRHS, sc_output), {});

    for (int h = 0; h < NRHS; ++h) {
        for (int r = 0; r < NR; ++r) {
            const auto ref = dr_output[h][r];
            EXPECT_FLOAT_EQ(ref, dc_output[h][r]);
            EXPECT_FLOAT_EQ(ref, sr_output[h][r]);
            EXPECT_FLOAT_EQ(ref, sc_output[h][r]);
        }
    }

    // Checking the transpose.
    std::vector<std::vector<double> > tr_output;
    auto transposed = std::make_unique<tatami::DenseColumnMatrix<double, int> >(NC, NR, dump);
    tatami_mult::multiply_with_multiple_vectors(rhs_ptrs, *transposed, formulate_ptrs(NR, NRHS, tr_output), {});
    for (int h = 0; h < NRHS; ++h) {
        EXPECT_EQ(tr_output[h], dr_output[h]);
    }
}

TEST(MultipleVectorsDispatch, Options) {
    tatami_mult::MultiplyWithMultipleVectorsOptions opt;
    tatami_mult::set_num_threads(opt, 13);
    EXPECT_EQ(opt.dense_row.num_threads, 13);
    EXPECT_EQ(opt.dense_column.num_threads, 13);
    EXPECT_EQ(opt.sparse_row.num_threads, 13);
    EXPECT_EQ(opt.sparse_column.num_threads, 13);

    tatami_mult::set_dense_primary_block_size(opt, 42);
    EXPECT_EQ(opt.dense_row.primary_block_size, 42);
    EXPECT_EQ(opt.dense_column.primary_block_size, 42);

    tatami_mult::set_dense_secondary_block_size(opt, 69);
    EXPECT_EQ(opt.dense_row.secondary_block_size, 69);
    EXPECT_EQ(opt.dense_column.secondary_block_size, 69);

    tatami_mult::set_sparse_block_size(opt, 42);
    EXPECT_EQ(opt.sparse_row.block_size, 42);
    EXPECT_EQ(opt.sparse_column.block_size, 42);
}

TEST(MultipleVectorsDispatch, Unsigned) {
    // Check that there are no compiler warnings when the indices are unsigned.
    // This aims to avoid the use of the wrong integer type in, e.g., comparisons. 
    const int NR = 43;
    const int NC = 92;
    const int NRHS = 21;

    auto dump = tatami_test::simulate_vector<double>(NR * NC, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.density = 0.24;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 12 + NR + NC + NRHS; 
        return opt;
    }());
    auto lhs = std::make_unique<tatami::DenseRowMatrix<double, int> >(NR, NC, dump);
    auto lhs_u = std::make_unique<tatami::DenseRowMatrix<double, unsigned> >(NR, NC, dump);

    auto rhs = tatami_test::simulate_vector<double>(NC * NRHS, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 50 + NR + NC + NRHS;
        return opt;
    }());

    std::vector<double*> rhs_ptrs(NRHS);
    for (int h = 0; h < NRHS; ++h) {
        rhs_ptrs[h] = rhs.data() + h * NC;
    }

    std::vector<std::vector<double> > ref, output;
    tatami_mult::multiply_with_multiple_vectors(*lhs, rhs_ptrs, formulate_ptrs(NR, NRHS, ref), {});
    tatami_mult::multiply_with_multiple_vectors(*lhs_u, rhs_ptrs, formulate_ptrs(NR, NRHS, output), {});
    EXPECT_EQ(ref, output);
}
