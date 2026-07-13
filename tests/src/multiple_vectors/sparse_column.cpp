#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "tatami_test/tatami_test.hpp"

#include "tatami_mult/multiple_vectors/sparse_column.hpp"

class MultipleVectorsSparseColumnTest : public ::testing::TestWithParam<std::tuple<int, int, int, int, int> > {};

TEST_P(MultipleVectorsSparseColumnTest, Basic) {
    const auto params = GetParam();
    const int NR = std::get<0>(params);
    const int NC = std::get<1>(params);
    const int NRHS = std::get<2>(params);
    const auto block_size = std::get<3>(params);
    const auto nthreads = std::get<4>(params);

    auto dump = tatami_test::simulate_vector<double>(NR * NC, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.density = 0.3;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 67 + NR + NC + block_size + nthreads;
        return opt;
    }());
    auto sparse_row = tatami::convert_to_compressed_sparse<double, int>(tatami::DenseRowMatrix<double, int>(NR, NC, dump), true, {});
    auto sparse_col = tatami::convert_to_compressed_sparse<double, int>(*sparse_row, false, {});

    auto rhs = tatami_test::simulate_vector<double>(NC * NRHS, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 39 + NR + NC + block_size + nthreads;
        return opt;
    }());

    std::vector<double*> rhs_ptrs(NRHS);
    for (int h = 0; h < NRHS; ++h) {
        rhs_ptrs[h] = rhs.data() + h * NC;
    }

    tatami_mult::MultiplySparseColumnWithMultipleVectorsOptions opt;
    opt.num_threads = nthreads;
    opt.block_size = block_size;

    std::vector<std::vector<double> > sr_output, sc_output;
    auto formulate_ptrs = [&](std::vector<std::vector<double> >& output) -> std::vector<double*> {
        output.resize(NRHS);
        std::vector<double*> ptrs(NRHS);
        for (int h = 0; h < NRHS; ++h) {
            // Setting an initial value for the output vectors, to check that dirty outputs are properly zeroed.
            output[h].resize(NR, 239 + h);
            ptrs[h] = output[h].data();
        }
        return ptrs;
    };

    tatami_mult::multiply_sparse_column_with_multiple_vectors(*sparse_row, rhs_ptrs, formulate_ptrs(sr_output), opt);
    tatami_mult::multiply_sparse_column_with_multiple_vectors(*sparse_col, rhs_ptrs, formulate_ptrs(sc_output), opt);

    for (int h = 0; h < NRHS; ++h) {
        for (int r = 0; r < NR; ++r) {
            const auto ref = std::inner_product(rhs_ptrs[h], rhs_ptrs[h] + NC, dump.begin() + r * NC, 0.0);
            EXPECT_FLOAT_EQ(ref, sr_output[h][r]);
            EXPECT_FLOAT_EQ(ref, sc_output[h][r]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MultipleVectors,
    MultipleVectorsSparseColumnTest,
    ::testing::Combine(
        ::testing::Values(100, 33), // number of rows.
        ::testing::Values(59, 148), // number of columns.
        ::testing::Values(20, 74),  // number of RHS vectors.
        ::testing::Values(1, 4, 8), // block size.
        ::testing::Values(1, 3)
    )
);
