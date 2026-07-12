#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "tatami_test/tatami_test.hpp"

#include "tatami_mult/multiple_vectors/dispatch.hpp"

class MultipleVectorsDispatchTest : public ::testing::TestWithParam<std::tuple<int, int, int, int> > {};

TEST_P(MultipleVectorsDispatchTest, Basic) {
    const auto params = GetParam();
    const int NR = std::get<0>(params);
    const int NC = std::get<1>(params);
    const int NRHS = std::get<2>(params);
    const auto nthreads = std::get<3>(params);

    auto dump = tatami_test::simulate_vector<double>(NR * NC, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.density = 0.24;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 77 + NR + NC + NRHS + nthreads;
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
        opt.seed = 50 + NR + NC + NRHS + nthreads;
        return opt;
    }());

    std::vector<double*> rhs_ptrs(NRHS);
    for (int h = 0; h < NRHS; ++h) {
        rhs_ptrs[h] = rhs.data() + h * NC;
    }

    auto formulate_ptrs = [&](std::vector<std::vector<double> >& output) -> std::vector<double*> {
        output.resize(NRHS);
        std::vector<double*> ptrs(NRHS);
        for (int h = 0; h < NRHS; ++h) {
            output[h].resize(NR);
            ptrs[h] = output[h].data();
        }
        return ptrs;
    };

    tatami_mult::MultiplyWithMultipleVectorsOptions opt;
    opt.dense_row.num_threads = nthreads;
    opt.dense_column.num_threads = nthreads;
    opt.sparse_row.num_threads = nthreads;
    opt.sparse_column.num_threads = nthreads;

    std::vector<std::vector<double> > dr_output, dc_output, sr_output, sc_output;
    tatami_mult::multiply_with_multiple_vectors(*dense_row, rhs_ptrs, formulate_ptrs(dr_output), opt);
    tatami_mult::multiply_with_multiple_vectors(*dense_col, rhs_ptrs, formulate_ptrs(dc_output), opt);
    tatami_mult::multiply_with_multiple_vectors(*sparse_row, rhs_ptrs, formulate_ptrs(sr_output), opt);
    tatami_mult::multiply_with_multiple_vectors(*sparse_col, rhs_ptrs, formulate_ptrs(sc_output), opt);

    for (int h = 0; h < NRHS; ++h) {
        for (int r = 0; r < NR; ++r) {
            const auto ref = dr_output[h][r];
            EXPECT_FLOAT_EQ(ref, dc_output[h][r]);
            EXPECT_FLOAT_EQ(ref, sr_output[h][r]);
            EXPECT_FLOAT_EQ(ref, sc_output[h][r]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MultipleVectors,
    MultipleVectorsDispatchTest,
    ::testing::Combine(
        ::testing::Values(107, 55),
        ::testing::Values(36, 78),
        ::testing::Values(13, 60),
        ::testing::Values(1, 3)
    )
);
