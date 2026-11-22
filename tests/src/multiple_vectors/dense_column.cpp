#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "tatami_test/tatami_test.hpp"

#include "tatami_mult/multiple_vectors/dense_column.hpp"

class MultipleVectorsDenseColumnTest : public ::testing::TestWithParam<std::tuple<int, int, int, std::pair<int, int>, int> > {};

TEST_P(MultipleVectorsDenseColumnTest, Basic) {
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

    std::vector<double*> rhs_ptrs(NRHS);
    for (int h = 0; h < NRHS; ++h) {
        rhs_ptrs[h] = rhs.data() + h * NC;
    }

    tatami_mult::MultiplyDenseColumnWithMultipleVectorsOptions opt;
    opt.num_threads = nthreads;
    opt.primary_block_size = blocks.first;
    opt.secondary_block_size = blocks.second;

    auto formulate_ptrs = [&](std::vector<std::vector<double> >& output) -> std::vector<double*> {
        output.resize(NRHS);
        std::vector<double*> ptrs(NRHS);
        for (int h = 0; h < NRHS; ++h) {
            // Setting an initial value for the output vectors, to check that dirty outputs are properly zeroed.
            output[h].resize(NR, 91 + h);
            ptrs[h] = output[h].data();
        }
        return ptrs;
    };

    std::vector<std::vector<double> > dr_output, dc_output; 
    tatami_mult::multiply_dense_column_with_multiple_vectors(*dense_row, rhs_ptrs, formulate_ptrs(dr_output), opt);
    tatami_mult::multiply_dense_column_with_multiple_vectors(*dense_col, rhs_ptrs, formulate_ptrs(dc_output), opt);

    // Check the overload with custom RHS/output functions.
    std::vector<std::vector<double> > dr_output2(NRHS), dc_output2(NRHS); 
    for (int h = 0; h < NRHS; ++h) {
        dr_output2[h].resize(NR, 44 + h);
        dc_output2[h].resize(NR, 1321 + h);
    }
    tatami_mult::multiply_dense_column_with_multiple_vectors(
        *dense_row,
        NRHS,
        [&](const int h) -> const double* { return rhs_ptrs[h]; },
        [&](const int h) -> double* { return dr_output2[h].data(); },
        opt
    );
    tatami_mult::multiply_dense_column_with_multiple_vectors(
        *dense_col,
        NRHS,
        [&](const int h) -> const double* { return rhs_ptrs[h]; },
        [&](const int h) -> double* { return dc_output2[h].data(); },
        opt
    );

    // Checking the values.
    for (int h = 0; h < NRHS; ++h) {
        for (int r = 0; r < NR; ++r) {
            const auto ref = std::inner_product(rhs_ptrs[h], rhs_ptrs[h] + NC, dump.begin() + r * NC, 0.0);
            EXPECT_FLOAT_EQ(ref, dr_output[h][r]);
            EXPECT_FLOAT_EQ(ref, dc_output[h][r]);
            EXPECT_FLOAT_EQ(ref, dr_output2[h][r]);
            EXPECT_FLOAT_EQ(ref, dc_output2[h][r]);
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

/*********************************/

TEST(MultipleVectorsDenseColumn, UnsignedFunctor) {
    // Check that the functor overloads are compiled without warnings for different combinations of integer signedness.
    const int NR = 79;
    const int NC = 123;
    const int NRHS = 35;

    auto dump = tatami_test::simulate_vector<double>(NR * NC, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 69 + NR + NC + NRHS; 
        return opt;
    }());
    tatami::DenseColumnMatrix<double, int> lhs(NR, NC, dump);
    tatami::DenseColumnMatrix<double, unsigned> lhs_u(NR, NC, dump);

    auto rhs = tatami_test::simulate_vector<double>(NC * NRHS, [&]{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 42 + NR + NC + NRHS; 
        return opt;
    }());
    auto get_right = [&](const int h) -> const double* { return rhs.data() + h * NC; };

    std::vector<std::vector<double> > ref(NRHS), out_us(NRHS), out_su(NRHS), out_uu(NRHS); 
    for (int h = 0; h < NRHS; ++h) {
        ref[h].resize(NR);
        out_us[h].resize(NR);
        out_su[h].resize(NR);
        out_uu[h].resize(NR);
    }

    tatami_mult::multiply_dense_column_with_multiple_vectors(lhs, NRHS, get_right, [&](const int h) -> double* { return ref[h].data(); }, {});
    tatami_mult::multiply_dense_column_with_multiple_vectors(lhs, (unsigned)NRHS, get_right, [&](const int h) -> double* { return out_su[h].data(); }, {});
    EXPECT_EQ(ref, out_su);

    tatami_mult::multiply_dense_column_with_multiple_vectors(lhs_u, NRHS, get_right, [&](const int h) -> double* { return out_us[h].data(); }, {});
    EXPECT_EQ(ref, out_us);

    tatami_mult::multiply_dense_column_with_multiple_vectors(lhs_u, (unsigned)NRHS, get_right, [&](const int h) -> double* { return out_uu[h].data(); }, {});
    EXPECT_EQ(ref, out_uu);
}
