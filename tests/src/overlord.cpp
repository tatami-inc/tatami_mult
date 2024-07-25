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
        std::vector<double> dump = tatami_test::simulate_sparse_vector<double>(NR * NC, 0.2, /* lower = */ -10, /* upper = */ 10, /* seed = */ 99);
        dense_row.reset(new tatami::DenseRowMatrix<double, int>(NR, NC, std::move(dump)));
        dense_column = tatami::convert_to_dense(dense_row.get(), false);
        sparse_row = tatami::convert_to_compressed_sparse(dense_row.get(), true);
        sparse_column = tatami::convert_to_compressed_sparse(dense_row.get(), false);
    }
};

TEST_F(OverlordTest, RightVector) {
    std::vector<double> vec = tatami_test::simulate_dense_vector<double>(NC, /* lower = */ -10, /* upper = */ 10, /* seed = */ 69);
    tatami_mult::Options opt;

    std::vector<double> output_dr(NR);
    tatami_mult::multiply(*dense_row, vec.data(), output_dr.data(), opt);

    std::vector<double> output_dc(NR);
    tatami_mult::multiply(*dense_column, vec.data(), output_dc.data(), opt);
    EXPECT_EQ(output_dr, output_dc);

    std::vector<double> output_sr(NR);
    tatami_mult::multiply(*sparse_row, vec.data(), output_sr.data(), opt);
    EXPECT_EQ(output_dr, output_sr);

    std::vector<double> output_sc(NR);
    tatami_mult::multiply(*sparse_column, vec.data(), output_sc.data(), opt);
    EXPECT_EQ(output_dr, output_sc);
}

TEST_F(OverlordTest, LeftVector) {
    std::vector<double> vec = tatami_test::simulate_dense_vector<double>(NR, /* lower = */ -10, /* upper = */ 10, /* seed = */ 70);
    tatami_mult::Options opt;

    std::vector<double> output_dr(NC);
    tatami_mult::multiply(vec.data(), *dense_row, output_dr.data(), opt);

    std::vector<double> output_dc(NC);
    tatami_mult::multiply(vec.data(), *dense_column, output_dc.data(), opt);
    EXPECT_EQ(output_dr, output_dc);

    std::vector<double> output_sr(NC);
    tatami_mult::multiply(vec.data(), *sparse_row, output_sr.data(), opt);
    EXPECT_EQ(output_dr, output_sr);

    std::vector<double> output_sc(NC);
    tatami_mult::multiply(vec.data(), *sparse_column, output_sc.data(), opt);
    EXPECT_EQ(output_dr, output_sc);
}

TEST_F(OverlordTest, RightVectors) {
    std::vector<double> vec = tatami_test::simulate_dense_vector<double>(NC * 5, /* lower = */ -10, /* upper = */ 10, /* seed = */ 71);
    auto vec_ptrs = populate_pointers(vec, NC, 5);
    tatami_mult::Options opt;

    std::vector<double> ref(NR * 5);
    {
        auto ref_ptrs = populate_pointers(ref, NR, 5);
        for (size_t i = 0; i < ref_ptrs.size(); ++i) {
            tatami_mult::multiply(*dense_row, vec_ptrs[i], ref_ptrs[i], opt);
        }
    }

    std::vector<double> output_dr(NR * 5);
    tatami_mult::multiply(*dense_row, vec_ptrs, populate_pointers(output_dr, NR, 5), opt);
    EXPECT_EQ(ref, output_dr);

    std::vector<double> output_dc(NR * 5);
    tatami_mult::multiply(*dense_column, vec_ptrs, populate_pointers(output_dc, NR, 5), opt);
    EXPECT_EQ(ref, output_dc);

    std::vector<double> output_sr(NR * 5);
    tatami_mult::multiply(*sparse_row, vec_ptrs, populate_pointers(output_sr, NR, 5), opt);
    EXPECT_EQ(ref, output_sr);

    std::vector<double> output_sc(NR * 5);
    tatami_mult::multiply(*sparse_column, vec_ptrs, populate_pointers(output_sc, NR, 5), opt);
    EXPECT_EQ(ref, output_sc);
}

TEST_F(OverlordTest, LeftVectors) {
    std::vector<double> vec = tatami_test::simulate_dense_vector<double>(NR * 5, /* lower = */ -10, /* upper = */ 10, /* seed = */ 71);
    auto vec_ptrs = populate_pointers(vec, NR, 5);
    tatami_mult::Options opt;

    std::vector<double> ref(NC * 5);
    {
        auto ref_ptrs = populate_pointers(ref, NC, 5);
        for (size_t i = 0; i < ref_ptrs.size(); ++i) {
            tatami_mult::multiply(vec_ptrs[i], *dense_row, ref_ptrs[i], opt);
        }
    }

    std::vector<double> output_dr(NC * 5);
    tatami_mult::multiply(vec_ptrs, *dense_row, populate_pointers(output_dr, NC, 5), opt);
    EXPECT_EQ(ref, output_dr);

    std::vector<double> output_dc(NC * 5);
    tatami_mult::multiply(vec_ptrs, *dense_column, populate_pointers(output_dc, NC, 5), opt);
    EXPECT_EQ(ref, output_dc);

    std::vector<double> output_sr(NC * 5);
    tatami_mult::multiply(vec_ptrs, *sparse_row, populate_pointers(output_sr, NC, 5), opt);
    EXPECT_EQ(ref, output_sr);

    std::vector<double> output_sc(NC * 5);
    tatami_mult::multiply(vec_ptrs, *sparse_column, populate_pointers(output_sc, NC, 5), opt);
    EXPECT_EQ(ref, output_sc);
}

TEST_F(OverlordTest, RightMatrixDense) {
    std::shared_ptr<tatami::Matrix<double, int> > rhs;
    std::vector<double> vec = tatami_test::simulate_dense_vector<double>(NC * 5, /* lower = */ -10, /* upper = */ 10, /* seed = */ 71);
    rhs.reset(new tatami::DenseColumnMatrix<double, int>(NC, 5, vec));

    tatami_mult::Options opt;
    opt.prefer_larger = false;

    std::vector<double> ref(NR * 5);
    tatami_mult::multiply(*dense_row, populate_pointers(vec, NC, 5), populate_pointers(ref, NR, 5), opt);

    std::vector<double> output_dr(NR * 5);
    tatami_mult::multiply(*dense_row, *rhs, output_dr.data(), opt);
    EXPECT_EQ(ref, output_dr);

    std::vector<double> output_dc(NR * 5);
    tatami_mult::multiply(*dense_column, *rhs, output_dc.data(), opt);
    EXPECT_EQ(ref, output_dc);

    std::vector<double> output_sr(NR * 5);
    tatami_mult::multiply(*sparse_row, *rhs, output_sr.data(), opt);
    EXPECT_EQ(ref, output_sr);

    std::vector<double> output_sc(NR * 5);
    tatami_mult::multiply(*sparse_column, *rhs, output_sc.data(), opt);
    EXPECT_EQ(ref, output_sc);
}

TEST_F(OverlordTest, LeftMatrixDense) {
    std::shared_ptr<tatami::Matrix<double, int> > lhs;
    std::vector<double> vec = tatami_test::simulate_dense_vector<double>(NR * 5, /* lower = */ -10, /* upper = */ 10, /* seed = */ 71);
    lhs.reset(new tatami::DenseRowMatrix<double, int>(5, NR, vec));

    tatami_mult::Options opt;
    opt.prefer_larger = false;

    std::vector<double> tref(NC * 5);
    tatami_mult::multiply(populate_pointers(vec, NR, 5), *dense_row, populate_pointers(tref, NC, 5), opt);
    std::vector<double> ref(NC * 5); 
    tatami::transpose(tref.data(), 5, NC, ref.data()); // need to transpose back to column-major for comparison.

    std::vector<double> output_dr(NC * 5);
    tatami_mult::multiply(*lhs, *dense_row, output_dr.data(), opt);
    EXPECT_EQ(ref, output_dr);

    std::vector<double> output_dc(NC * 5);
    tatami_mult::multiply(*lhs, *dense_column, output_dc.data(), opt);
    EXPECT_EQ(ref, output_dc);

    std::vector<double> output_sr(NC * 5);
    tatami_mult::multiply(*lhs, *sparse_row, output_sr.data(), opt);
    EXPECT_EQ(ref, output_sr);

    std::vector<double> output_sc(NC * 5);
    tatami_mult::multiply(*lhs, *sparse_column, output_sc.data(), opt);
    EXPECT_EQ(ref, output_sc);
}

TEST_F(OverlordTest, RightMatrixSparse) {
    std::vector<double> vec = tatami_test::simulate_dense_vector<double>(NC * 5, /* lower = */ -10, /* upper = */ 10, /* seed = */ 71);
    std::shared_ptr<tatami::Matrix<double, int> > rhs;
    {
        tatami::DenseColumnMatrix<double, int> dense(NC, 5, vec);
        rhs = tatami::convert_to_compressed_sparse<double>(&dense, true);
    }

    tatami_mult::Options opt;
    opt.prefer_larger = false;

    std::vector<double> ref(NR * 5);
    tatami_mult::multiply(*dense_row, populate_pointers(vec, NC, 5), populate_pointers(ref, NR, 5), opt);

    std::vector<double> output_dr(NR * 5);
    tatami_mult::multiply(*dense_row, *rhs, output_dr.data(), opt);
    EXPECT_EQ(ref, output_dr);

    std::vector<double> output_dc(NR * 5);
    tatami_mult::multiply(*dense_column, *rhs, output_dc.data(), opt);
    EXPECT_EQ(ref, output_dc);

    std::vector<double> output_sr(NR * 5);
    tatami_mult::multiply(*sparse_row, *rhs, output_sr.data(), opt);
    EXPECT_EQ(ref, output_sr);

    std::vector<double> output_sc(NR * 5);
    tatami_mult::multiply(*sparse_column, *rhs, output_sc.data(), opt);
    EXPECT_EQ(ref, output_sc);
}

TEST_F(OverlordTest, LeftMatrixSparse) {
    std::vector<double> vec = tatami_test::simulate_dense_vector<double>(NR * 5, /* lower = */ -10, /* upper = */ 10, /* seed = */ 71);
    std::shared_ptr<tatami::Matrix<double, int> > lhs;
    {
        tatami::DenseRowMatrix<double, int> dense(5, NR, vec);
        lhs = tatami::convert_to_compressed_sparse<double>(&dense, true);
    }

    tatami_mult::Options opt;
    opt.prefer_larger = false;

    std::vector<double> tref(NC * 5);
    tatami_mult::multiply(populate_pointers(vec, NR, 5), *dense_row, populate_pointers(tref, NC, 5), opt);
    std::vector<double> ref(NC * 5);
    tatami::transpose(tref.data(), 5, NC, ref.data()); // need to transpose back to column-major for comparison.

    std::vector<double> output_dr(NC * 5);
    tatami_mult::multiply(*lhs, *dense_row, output_dr.data(), opt);
    EXPECT_EQ(ref, output_dr);

    std::vector<double> output_dc(NC * 5);
    tatami_mult::multiply(*lhs, *dense_column, output_dc.data(), opt);
    EXPECT_EQ(ref, output_dc);

    std::vector<double> output_sr(NC * 5);
    tatami_mult::multiply(*lhs, *sparse_row, output_sr.data(), opt);
    EXPECT_EQ(ref, output_sr);

    std::vector<double> output_sc(NC * 5);
    tatami_mult::multiply(*lhs, *sparse_column, output_sc.data(), opt);
    EXPECT_EQ(ref, output_sc);
}

TEST_F(OverlordTest, MatrixOptions) {
    std::shared_ptr<tatami::Matrix<double, int> > lhs;
    {
        std::vector<double> vec = tatami_test::simulate_dense_vector<double>(NR * 5, /* lower = */ -10, /* upper = */ 10, /* seed = */ 71);
        lhs.reset(new tatami::DenseRowMatrix<double, int>(5, NR, std::move(vec)));
    }

    tatami_mult::Options opt;
    opt.prefer_larger = false;
    std::vector<double> ref(NC * 5);
    tatami_mult::multiply(*lhs, *dense_row, ref.data(), opt);

    // Checking that the switch is done correctly when we allow it to choose the iteration order.
    {
        tatami_mult::Options opt;
        opt.prefer_larger = true;
        std::vector<double> out(NC * 5);
        tatami_mult::multiply(*lhs, *dense_row, out.data(), opt);
        EXPECT_EQ(ref, out);
    }

    // Checking that we can save in row-major order.
    {
        tatami_mult::Options opt;
        opt.prefer_larger = false;
        opt.column_major_output = false;
        std::vector<double> tout(NC * 5);
        tatami_mult::multiply(*lhs, *dense_row, tout.data(), opt);
        std::vector<double> out(NC * 5);
        tatami::transpose(tout.data(), 5, NC, out.data());
        EXPECT_EQ(ref, out);
    }

    // Checking that we can save in row-major order, combined with automatic choice of iteration order.
    {
        tatami_mult::Options opt;
        opt.column_major_output = false;
        std::vector<double> tout(NC * 5);
        tatami_mult::multiply(*lhs, *dense_row, tout.data(), opt);
        std::vector<double> out(NC * 5);
        tatami::transpose(tout.data(), 5, NC, out.data());
        EXPECT_EQ(ref, out);
    }
}
