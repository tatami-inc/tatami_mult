#include <gtest/gtest.h>

#include "tatami_mult/tatami_mult.hpp"
#include "tatami_test/tatami_test.hpp"

#include "utils.h"

class OverlordTest : public ::testing::Test {
protected:
    inline static size_t NR, NC;
    inline static std::shared_ptr<tatami::Matrix<double, int> > dense_row, dense_column, sparse_row, sparse_column;

    static void SetUpTestSuite() {
        NR = 82;
        NC = 52;

        std::vector<double> dump = tatami_test::simulate_vector<double>(NR * NC, []{
            tatami_test::SimulateVectorOptions opt;
            opt.density = 0.2;
            opt.lower = -10;
            opt.upper = 10;
            opt.seed = 99;
            return opt;
        }());

        dense_row.reset(new tatami::DenseRowMatrix<double, int>(NR, NC, std::move(dump)));
        dense_column = tatami::convert_to_dense(dense_row.get(), false);
        sparse_row = tatami::convert_to_compressed_sparse(dense_row.get(), true);
        sparse_column = tatami::convert_to_compressed_sparse(dense_row.get(), false);
    }

    std::vector<double*> populate_pointers(std::vector<double>& src, std::size_t stride, std::size_t num) {
        std::vector<double*> output(num);
        for (std::size_t i = 0; i < num; ++i) {
            output[i] = src.data() + i * stride;
        }
        return output;
    }
};

TEST_F(OverlordTest, RightVector) {
    std::vector<double> vec = tatami_test::simulate_vector<double>(NC, []{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 69;
        return opt;
    }());
    tatami_mult::Options opt;

    std::vector<double> output_dr(NR);
    tatami_mult::multiply(*dense_row, vec.data(), output_dr.data(), opt);

    std::vector<double> output_dc(NR);
    tatami_mult::multiply(*dense_column, vec.data(), output_dc.data(), opt);

    std::vector<double> output_sr(NR);
    tatami_mult::multiply(*sparse_row, vec.data(), output_sr.data(), opt);

    std::vector<double> output_sc(NR);
    tatami_mult::multiply(*sparse_column, vec.data(), output_sc.data(), opt);

    for (std::size_t r = 0; r < NR; ++r) {
        EXPECT_FLOAT_EQ(output_dr[r], output_sc[r]);
        EXPECT_FLOAT_EQ(output_dr[r], output_dc[r]);
        EXPECT_FLOAT_EQ(output_dr[r], output_sr[r]);
    }
}

TEST_F(OverlordTest, LeftVector) {
    std::vector<double> vec = tatami_test::simulate_vector<double>(NR, []{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 70;
        return opt;
    }());
    tatami_mult::Options opt;

    std::vector<double> output_dr(NC);
    tatami_mult::multiply(vec.data(), *dense_row, output_dr.data(), opt);

    std::vector<double> output_dc(NC);
    tatami_mult::multiply(vec.data(), *dense_column, output_dc.data(), opt);

    std::vector<double> output_sr(NC);
    tatami_mult::multiply(vec.data(), *sparse_row, output_sr.data(), opt);

    std::vector<double> output_sc(NC);
    tatami_mult::multiply(vec.data(), *sparse_column, output_sc.data(), opt);

    for (std::size_t c = 0; c < NC; ++c) {
        EXPECT_FLOAT_EQ(output_dr[c], output_sc[c]);
        EXPECT_FLOAT_EQ(output_dr[c], output_dc[c]);
        EXPECT_FLOAT_EQ(output_dr[c], output_sr[c]);
    }
}

TEST_F(OverlordTest, RightVectors) {
    std::vector<double> vec = tatami_test::simulate_vector<double>(NC * 5, []{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 71;
        return opt;
    }());
    auto vec_ptrs = populate_pointers(vec, NC, 5);
    tatami_mult::Options opt;

    const std::size_t full_size = NR * 5; 
    std::vector<double> ref(full_size);
    {
        auto ref_ptrs = populate_pointers(ref, NR, 5);
        for (size_t i = 0; i < ref_ptrs.size(); ++i) {
            tatami_mult::multiply(*dense_row, vec_ptrs[i], ref_ptrs[i], opt);
        }
    }

    std::vector<double> output_dr(full_size);
    tatami_mult::multiply(*dense_row, vec_ptrs, populate_pointers(output_dr, NR, 5), opt);

    std::vector<double> output_dc(full_size);
    tatami_mult::multiply(*dense_column, vec_ptrs, populate_pointers(output_dc, NR, 5), opt);

    std::vector<double> output_sr(full_size);
    tatami_mult::multiply(*sparse_row, vec_ptrs, populate_pointers(output_sr, NR, 5), opt);

    std::vector<double> output_sc(full_size);
    tatami_mult::multiply(*sparse_column, vec_ptrs, populate_pointers(output_sc, NR, 5), opt);

    for (std::size_t f = 0; f < full_size; ++f) {
        EXPECT_FLOAT_EQ(ref[f], output_dr[f]);
        EXPECT_FLOAT_EQ(ref[f], output_sc[f]);
        EXPECT_FLOAT_EQ(ref[f], output_dc[f]);
        EXPECT_FLOAT_EQ(ref[f], output_sr[f]);
    }
}

TEST_F(OverlordTest, LeftVectors) {
    std::vector<double> vec = tatami_test::simulate_vector<double>(NR * 5, []{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 72;
        return opt;
    }());
    auto vec_ptrs = populate_pointers(vec, NR, 5);
    tatami_mult::Options opt;

    const std::size_t full_size = NC * 5; 
    std::vector<double> ref(full_size);
    {
        auto ref_ptrs = populate_pointers(ref, NC, 5);
        for (size_t i = 0; i < ref_ptrs.size(); ++i) {
            tatami_mult::multiply(vec_ptrs[i], *dense_row, ref_ptrs[i], opt);
        }
    }

    std::vector<double> output_dr(full_size);
    tatami_mult::multiply(vec_ptrs, *dense_row, populate_pointers(output_dr, NC, 5), opt);

    std::vector<double> output_dc(full_size);
    tatami_mult::multiply(vec_ptrs, *dense_column, populate_pointers(output_dc, NC, 5), opt);

    std::vector<double> output_sr(full_size);
    tatami_mult::multiply(vec_ptrs, *sparse_row, populate_pointers(output_sr, NC, 5), opt);

    std::vector<double> output_sc(full_size);
    tatami_mult::multiply(vec_ptrs, *sparse_column, populate_pointers(output_sc, NC, 5), opt);

    for (std::size_t f = 0; f < full_size; ++f) {
        EXPECT_FLOAT_EQ(ref[f], output_dr[f]);
        EXPECT_FLOAT_EQ(ref[f], output_sc[f]);
        EXPECT_FLOAT_EQ(ref[f], output_dc[f]);
        EXPECT_FLOAT_EQ(ref[f], output_sr[f]);
    }
}

TEST_F(OverlordTest, RightMatrixDense) {
    std::shared_ptr<tatami::Matrix<double, int> > rhs;
    std::vector<double> vec = tatami_test::simulate_vector<double>(NC * 5, []{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 73;
        return opt;
    }());
    rhs.reset(new tatami::DenseColumnMatrix<double, int>(NC, 5, vec));

    tatami_mult::Options opt;
    opt.prefer_larger = false;

    const std::size_t full_size = NR * 5; 
    std::vector<double> ref(full_size);
    tatami_mult::multiply(*dense_row, populate_pointers(vec, NC, 5), populate_pointers(ref, NR, 5), opt);

    std::vector<double> output_dr(full_size);
    tatami_mult::multiply(*dense_row, *rhs, output_dr.data(), opt);

    std::vector<double> output_dc(full_size);
    tatami_mult::multiply(*dense_column, *rhs, output_dc.data(), opt);

    std::vector<double> output_sr(full_size);
    tatami_mult::multiply(*sparse_row, *rhs, output_sr.data(), opt);

    std::vector<double> output_sc(full_size);
    tatami_mult::multiply(*sparse_column, *rhs, output_sc.data(), opt);

    for (std::size_t f = 0; f < full_size; ++f) {
        EXPECT_FLOAT_EQ(ref[f], output_dr[f]);
        EXPECT_FLOAT_EQ(ref[f], output_sc[f]);
        EXPECT_FLOAT_EQ(ref[f], output_dc[f]);
        EXPECT_FLOAT_EQ(ref[f], output_sr[f]);
    }
}

TEST_F(OverlordTest, LeftMatrixDense) {
    std::shared_ptr<tatami::Matrix<double, int> > lhs;
    std::vector<double> vec = tatami_test::simulate_vector<double>(NR * 5, []{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 74;
        return opt;
    }());
    lhs.reset(new tatami::DenseRowMatrix<double, int>(5, NR, vec));

    tatami_mult::Options opt;
    opt.prefer_larger = false;

    const std::size_t full_size = NR * 5; 
    std::vector<double> tref(full_size);
    tatami_mult::multiply(populate_pointers(vec, NR, 5), *dense_row, populate_pointers(tref, NC, 5), opt);
    std::vector<double> ref(full_size); 
    tatami::transpose(tref.data(), 5, NC, ref.data()); // need to transpose back to column-major for comparison.

    std::vector<double> output_dr(full_size);
    tatami_mult::multiply(*lhs, *dense_row, output_dr.data(), opt);

    std::vector<double> output_dc(full_size);
    tatami_mult::multiply(*lhs, *dense_column, output_dc.data(), opt);

    std::vector<double> output_sr(full_size);
    tatami_mult::multiply(*lhs, *sparse_row, output_sr.data(), opt);

    std::vector<double> output_sc(full_size);
    tatami_mult::multiply(*lhs, *sparse_column, output_sc.data(), opt);

    for (std::size_t f = 0; f < full_size; ++f) {
        EXPECT_FLOAT_EQ(ref[f], output_dr[f]);
        EXPECT_FLOAT_EQ(ref[f], output_sc[f]);
        EXPECT_FLOAT_EQ(ref[f], output_dc[f]);
        EXPECT_FLOAT_EQ(ref[f], output_sr[f]);
    }
}

TEST_F(OverlordTest, RightMatrixSparse) {
    std::vector<double> vec = tatami_test::simulate_vector<double>(NC * 5, []{
        tatami_test::SimulateVectorOptions opt;
        opt.density = 0.1;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 75;
        return opt;
    }());

    std::shared_ptr<tatami::Matrix<double, int> > rhs;
    {
        tatami::DenseColumnMatrix<double, int> dense(NC, 5, vec);
        rhs = tatami::convert_to_compressed_sparse<double>(&dense, true);
    }

    tatami_mult::Options opt;
    opt.prefer_larger = false;

    const std::size_t full_size = NR * 5; 
    std::vector<double> ref(full_size);
    tatami_mult::multiply(*dense_row, populate_pointers(vec, NC, 5), populate_pointers(ref, NR, 5), opt);

    std::vector<double> output_dr(full_size);
    tatami_mult::multiply(*dense_row, *rhs, output_dr.data(), opt);

    std::vector<double> output_dc(full_size);
    tatami_mult::multiply(*dense_column, *rhs, output_dc.data(), opt);

    std::vector<double> output_sr(full_size);
    tatami_mult::multiply(*sparse_row, *rhs, output_sr.data(), opt);

    std::vector<double> output_sc(full_size);
    tatami_mult::multiply(*sparse_column, *rhs, output_sc.data(), opt);

    for (std::size_t f = 0; f < full_size; ++f) {
        EXPECT_FLOAT_EQ(ref[f], output_dr[f]);
        EXPECT_FLOAT_EQ(ref[f], output_sc[f]);
        EXPECT_FLOAT_EQ(ref[f], output_dc[f]);
        EXPECT_FLOAT_EQ(ref[f], output_sr[f]);
    }
}

TEST_F(OverlordTest, LeftMatrixSparse) {
    std::vector<double> vec = tatami_test::simulate_vector<double>(NR * 5, []{ 
        tatami_test::SimulateVectorOptions opt;
        opt.density = 0.1;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 76;
        return opt;
    }());

    std::shared_ptr<tatami::Matrix<double, int> > lhs;
    {
        tatami::DenseRowMatrix<double, int> dense(5, NR, vec);
        lhs = tatami::convert_to_compressed_sparse<double>(&dense, true);
    }

    tatami_mult::Options opt;
    opt.prefer_larger = false;

    const std::size_t full_size = NC * 5; 
    std::vector<double> tref(NC * 5);
    tatami_mult::multiply(populate_pointers(vec, NR, 5), *dense_row, populate_pointers(tref, NC, 5), opt);
    std::vector<double> ref(full_size);
    tatami::transpose(tref.data(), 5, NC, ref.data()); // need to transpose back to column-major for comparison.

    std::vector<double> output_dr(full_size);
    tatami_mult::multiply(*lhs, *dense_row, output_dr.data(), opt);

    std::vector<double> output_dc(full_size);
    tatami_mult::multiply(*lhs, *dense_column, output_dc.data(), opt);

    std::vector<double> output_sr(full_size);
    tatami_mult::multiply(*lhs, *sparse_row, output_sr.data(), opt);

    std::vector<double> output_sc(full_size);
    tatami_mult::multiply(*lhs, *sparse_column, output_sc.data(), opt);

    for (std::size_t f = 0; f < full_size; ++f) {
        EXPECT_FLOAT_EQ(ref[f], output_dr[f]);
        EXPECT_FLOAT_EQ(ref[f], output_sc[f]);
        EXPECT_FLOAT_EQ(ref[f], output_dc[f]);
        EXPECT_FLOAT_EQ(ref[f], output_sr[f]);
    }
}

TEST_F(OverlordTest, MatrixOptions) {
    std::shared_ptr<tatami::Matrix<double, int> > lhs;
    {
        std::vector<double> vec = tatami_test::simulate_vector<double>(NR * 5, []{
            tatami_test::SimulateVectorOptions opt;
            opt.lower = -10;
            opt.upper = 10;
            opt.seed = 77;
            return opt;
        }());
        lhs.reset(new tatami::DenseRowMatrix<double, int>(5, NR, std::move(vec)));
    }

    tatami_mult::Options opt;
    opt.prefer_larger = false;
    const std::size_t full_size = NC * 5; 
    std::vector<double> ref(full_size);
    tatami_mult::multiply(*lhs, *dense_row, ref.data(), opt);

    // Checking that the switch is done correctly when we allow it to choose the iteration order.
    {
        tatami_mult::Options opt;
        opt.prefer_larger = true;
        std::vector<double> out(full_size);
        tatami_mult::multiply(*lhs, *dense_row, out.data(), opt);
        for (std::size_t f = 0; f < full_size; ++f) {
            EXPECT_FLOAT_EQ(ref[f], out[f]);
        }
    }

    // Checking that we can save in row-major order.
    {
        tatami_mult::Options opt;
        opt.prefer_larger = false;
        opt.column_major_output = false;
        std::vector<double> tout(full_size);
        tatami_mult::multiply(*lhs, *dense_row, tout.data(), opt);
        std::vector<double> out(full_size);
        tatami::transpose(tout.data(), 5, NC, out.data());
        for (std::size_t f = 0; f < full_size; ++f) {
            EXPECT_FLOAT_EQ(ref[f], out[f]);
        }
    }

    // Checking that we can save in row-major order, combined with automatic choice of iteration order.
    {
        tatami_mult::Options opt;
        opt.column_major_output = false;
        std::vector<double> tout(full_size);
        tatami_mult::multiply(*lhs, *dense_row, tout.data(), opt);
        std::vector<double> out(full_size);
        tatami::transpose(tout.data(), 5, NC, out.data());
        for (std::size_t f = 0; f < full_size; ++f) {
            EXPECT_FLOAT_EQ(ref[f], out[f]);
        }
    }
}

TEST(MultiplyMatrix, Options) {
    tatami_mult::MultiplyWithMatrixOptions opt;
    tatami_mult::set_num_threads(opt, 13);
    EXPECT_EQ(opt.dense_matrix.dense_row.column_to_column.num_threads, 13);
    EXPECT_EQ(opt.sparse_matrix.dense_row.column_to_column.num_threads, 13);

    tatami_mult::set_dense_primary_block_size(opt, 100);
    EXPECT_EQ(opt.dense_matrix.dense_row.column_to_column.primary_block_size, 100);

    tatami_mult::set_dense_secondary_block_size(opt, 55);
    EXPECT_EQ(opt.dense_matrix.dense_row.column_to_column.secondary_block_size, 55);

    tatami_mult::set_sparse_block_size(opt, 100);
    EXPECT_EQ(opt.dense_matrix.sparse_row.column_to_column.block_size, 100);
    EXPECT_EQ(opt.sparse_matrix.sparse_row.column_to_column.block_size, 100);
}
