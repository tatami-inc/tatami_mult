#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "tatami_test/tatami_test.hpp"

#include "tatami_mult/multiple_vectors/dense_column/public.hpp"

class MultipleVectorsDenseColumnTest : public ::testing::TestWithParam<std::tuple<int, int, int, std::pair<int, int>, int> > {};

TEST_P(MultipleVectorsDenseColumnTest, Vector) {
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

    std::vector<double*> rhs_ptrs(NRHS);
    for (int h = 0; h < NRHS; ++h) {
        rhs_ptrs[h] = rhs.data() + h * NC;
    }

    tatami_mult::MultiplyDenseColumnWithMultipleVectorsOptions opt;
    opt.num_threads = nthreads;
    opt.primary_block_size = blocks.first;
    opt.secondary_block_size = blocks.second;

    std::vector<std::vector<double> > dr_output, dc_output; 
    auto formulate_ptrs = [&](std::vector<std::vector<double> >& output) -> std::vector<double*> {
        output.resize(NRHS);
        std::vector<double*> ptrs(NRHS);
        for (int h = 0; h < NRHS; ++h) {
            output[h].resize(NR);
            ptrs[h] = output[h].data();
        }
        return ptrs;
    };

    tatami_mult::multiply_dense_column_with_multiple_vectors(*dense_row, rhs_ptrs, formulate_ptrs(dr_output), opt);
    tatami_mult::multiply_dense_column_with_multiple_vectors(*dense_col, rhs_ptrs, formulate_ptrs(dc_output), opt);

    for (int h = 0; h < NRHS; ++h) {
        for (int r = 0; r < NR; ++r) {
            const auto ref = std::inner_product(rhs_ptrs[h], rhs_ptrs[h] + NC, dump.begin() + r * NC, 0.0);
            EXPECT_FLOAT_EQ(ref, dr_output[h][r]);
            EXPECT_FLOAT_EQ(ref, dc_output[h][r]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MultipleVectors,
    MultipleVectorsDenseColumnTest,
    ::testing::Combine(
        ::testing::Values(80, 45), // number of rows.
        ::testing::Values(53, 124), // number of columns.
        ::testing::Values(31, 88),  // number of RHS vectors.
        ::testing::Values(          // block size.
            std::make_pair(1, 0),
            std::make_pair(4, 16),
            std::make_pair(8, 8)
        ),
        ::testing::Values(1, 3)
    )
);
