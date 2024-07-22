#include <gtest/gtest.h>

#include "tatami_mult/sparse_row.hpp"
#include "tatami_test/tatami_test.hpp"

#include "utils.h"

TEST(SparseRow, Vector) {
    size_t NR = 99, NC = 152;
    auto dump = tatami_test::simulate_sparse_vector<double>(NR * NC, 0.1);
    tatami::DenseRowMatrix<double, int> raw_mat(NR, NC, dump);
    auto spptr = tatami::convert_to_compressed_sparse(&raw_mat, true);
    auto rhs = tatami_test::simulate_sparse_vector<double>(NC, /* lower = */ -10, /* upper = */ 10, /* seed = */ 421);

    // Doing a reference calculation.
    std::vector<double> ref(NR);
    for (size_t r = 0; r < NR; ++r) {
        ref[r] = std::inner_product(rhs.begin(), rhs.end(), dump.begin() + r * NC, 0.0);
    }

    std::vector<double> output(NR);
    tatami_mult::internal::sparse_row_vector(*spptr, rhs.data(), output.data(), 1);
    EXPECT_EQ(output, ref);

    // Same results with multiple threads.
    std::fill(output.begin(), output.end(), 0);
    tatami_mult::internal::sparse_row_vector(*spptr, rhs.data(), output.data(), 3);
    EXPECT_EQ(output, ref);

    // Correct results when we throw in an NaN.
    rhs[0] = std::numeric_limits<double>::quiet_NaN();
    tatami_mult::internal::sparse_row_vector(*spptr, rhs.data(), output.data(), 1);
    for (auto x : output) {
        EXPECT_TRUE(std::isnan(x));
    }
}

TEST(SparseRow, VectorNoSpecial) {
    size_t NR = 99, NC = 152;
    auto dump = tatami_test::simulate_sparse_vector<int>(NR * NC, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 69);
    tatami::DenseRowMatrix<int, int> raw_mat(NR, NC, dump);
    auto spptr = tatami::convert_to_compressed_sparse<int>(&raw_mat, true);
    auto rhs = tatami_test::simulate_sparse_vector<double>(NC, /* lower = */ -10, /* upper = */ 10, /* seed = */ 421);

    tatami::DenseRowMatrix<double, int> dmat(NR, NC, std::vector<double>(dump.begin(), dump.end()));
    std::vector<double> ref(NR);
    tatami_mult::internal::sparse_row_vector(dmat, rhs.data(), ref.data(), 1);

    std::vector<double> output(NR);
    tatami_mult::internal::sparse_row_vector(*spptr, rhs.data(), output.data(), 1);
    EXPECT_EQ(output, ref);
}

TEST(SparseRow, Matrix) {
    size_t NR = 129, NC = 60;
    auto dump = tatami_test::simulate_sparse_vector<double>(NR * NC, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 69);
    tatami::DenseRowMatrix<double, int> raw_mat(NR, NC, dump);
    auto spptr = tatami::convert_to_compressed_sparse(&raw_mat, true);
    auto rhs = tatami_test::simulate_dense_vector<double>(NC * 2, /* lower = */ -10, /* upper = */ 10, /* seed = */ 421);

    // Doing a reference calculation.
    std::vector<double> ref(NR * 2);
    tatami_mult::internal::sparse_row_vector(*spptr, rhs.data(), ref.data(), 1);
    tatami_mult::internal::sparse_row_vector(*spptr, rhs.data() + NC, ref.data() + NR, 1);

    std::vector<double> output(NR * 2);
    tatami_mult::internal::sparse_row_matrix(*spptr, rhs.data(), 2, output.data(), 1);
    EXPECT_EQ(output, ref);

    // Same results with multiple threads.
    std::fill(output.begin(), output.end(), 0);
    tatami_mult::internal::sparse_row_matrix(*spptr, rhs.data(), 2, output.data(), 3);
    EXPECT_EQ(output, ref);

    // Correct results when we throw in an NaN.
    rhs[0] = std::numeric_limits<double>::quiet_NaN();
    rhs[2 * NC - 1] = std::numeric_limits<double>::quiet_NaN();
    tatami_mult::internal::sparse_row_matrix(*spptr, rhs.data(), 2, output.data(), 1);
    for (auto x : output) {
        EXPECT_TRUE(std::isnan(x));
    }
}

TEST(SparseRow, MatrixNoSpecial) {
    size_t NR = 99, NC = 152;
    auto dump = tatami_test::simulate_sparse_vector<int>(NR * NC, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 69);
    tatami::DenseRowMatrix<int, int> raw_mat(NR, NC, dump);
    auto spptr = tatami::convert_to_compressed_sparse<int>(&raw_mat, true);
    auto rhs = tatami_test::simulate_sparse_vector<double>(NC * 2, /* lower = */ -10, /* upper = */ 10, /* seed = */ 421);

    tatami::DenseRowMatrix<double, int> dmat(NR, NC, std::vector<double>(dump.begin(), dump.end()));
    std::vector<double> ref(NR * 2);
    tatami_mult::internal::sparse_row_matrix(dmat, rhs.data(), 2, ref.data(), 1);

    std::vector<double> output(NR * 2);
    tatami_mult::internal::sparse_row_matrix(*spptr, rhs.data(), 2, output.data(), 1);
    EXPECT_EQ(output, ref);
}

TEST(SparseRow, TatamiDense) {
    size_t NR = 131, NC = 51;
    auto dump = tatami_test::simulate_sparse_vector<int>(NR * NC, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 69);
    tatami::DenseRowMatrix<int, int> raw_mat(NR, NC, dump);
    auto spptr = tatami::convert_to_compressed_sparse<int>(&raw_mat, true);

    auto rhs = tatami_test::simulate_sparse_vector<double>(NC * 2, /* lower = */ -10, /* upper = */ 10, /* seed = */ 421);
    tatami::DenseColumnMatrix<double, int> rhs_mat(NC, 2, rhs);

    // Doing a reference calculation.
    std::vector<double> ref(NR * 2);
    tatami_mult::internal::sparse_row_matrix(*spptr, rhs.data(), 2, ref.data(), 1);

    std::vector<double> output(NR * 2);
    tatami_mult::internal::sparse_row_tatami_dense(*spptr, rhs_mat, output.data(), 1);
    EXPECT_EQ(output, ref);

    // Same results with multiple threads.
    std::fill(output.begin(), output.end(), 0);
    tatami_mult::internal::sparse_row_tatami_dense(*spptr, rhs_mat, output.data(), 3);
    EXPECT_EQ(output, ref);

    // Correct results when we throw in an NaN.
    rhs[0] = std::numeric_limits<double>::quiet_NaN();
    rhs[2 * NC - 1] = std::numeric_limits<double>::quiet_NaN();
    rhs_mat = tatami::DenseColumnMatrix<double, int>(NC, 2, rhs);
    tatami_mult::internal::sparse_row_matrix(*spptr, rhs.data(), 2, output.data(), 1);
    for (auto x : output) {
        EXPECT_TRUE(std::isnan(x));
    }
}

TEST(SparseRow, TatamiDenseNoSpecial) {
    size_t NR = 131, NC = 51;
    auto dump = tatami_test::simulate_sparse_vector<int>(NR * NC, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 69);
    tatami::DenseRowMatrix<int, int> raw_mat(NR, NC, dump);
    auto spptr = tatami::convert_to_compressed_sparse<int>(&raw_mat, true);

    auto rhs = tatami_test::simulate_sparse_vector<double>(NC * 2, /* lower = */ -10, /* upper = */ 10, /* seed = */ 421);
    tatami::DenseColumnMatrix<double, int> rhs_mat(NC, 2, rhs);

    tatami::DenseRowMatrix<double, int> dmat(NR, NC, std::vector<double>(dump.begin(), dump.end()));
    std::vector<double> ref(NR * 2);
    tatami_mult::internal::sparse_row_tatami_dense(dmat, rhs_mat, ref.data(), 1);

    std::vector<double> output(NR * 2);
    tatami_mult::internal::sparse_row_tatami_dense(*spptr, rhs_mat, output.data(), 1);
    EXPECT_EQ(output, ref);
}

TEST(SparseRow, TatamiSparse) {
    size_t NR = 131, NC = 51;
    auto dump = tatami_test::simulate_sparse_vector<double>(NR * NC, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 69);
    tatami::DenseRowMatrix<double, int> raw_mat(NR, NC, dump);
    auto spptr = tatami::convert_to_compressed_sparse(&raw_mat, true);

    auto rhs = tatami_test::simulate_sparse_vector<double>(NC * 2, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 421);
    tatami::SparseColumnMatrix<double, int> rhs_mat(NC, 2, rhs);
    auto rhs_spmat = tatami::convert_to_compressed_sparse(&rhs_mat, false);

    // Doing a reference calculation.
    std::vector<double> ref(NR * 2);
    tatami_mult::internal::sparse_row_tatami_sparse(*spptr, rhs_mat, ref.data(), 1);

    std::vector<double> output(NR * 2);
    tatami_mult::internal::sparse_row_tatami_sparse(*spptr, *rhs_spmat, output.data(), 1);
    EXPECT_EQ(output, ref);

    // Same results with multiple threads.
    std::fill(output.begin(), output.end(), 0);
    tatami_mult::internal::dense_row_tatami_sparse(*spptr, *rhs_spmat, output.data(), 3);
    EXPECT_EQ(output, ref);

    // Correct results when we throw in an NaN.
    rhs[0] = std::numeric_limits<double>::quiet_NaN();
    rhs[2 * NC - 1] = std::numeric_limits<double>::quiet_NaN();
    rhs_mat = tatami::DenseColumnMatrix<double, int>(NC, 2, rhs);
    rhs_spmat = tatami::convert_to_compressed_sparse(&rhs_mat, false);
    tatami_mult::internal::sparse_row_matrix(*spptr, rhs.data(), 2, output.data(), 1);
    for (auto x : output) {
        EXPECT_TRUE(std::isnan(x));
    }
}

TEST(SparseRow, TatamiDenseNoSpecial) {
    size_t NR = 131, NC = 51;
    auto dump = tatami_test::simulate_sparse_vector<int>(NR * NC, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 69);
    tatami::DenseRowMatrix<int, int> raw_mat(NR, NC, dump);
    auto spptr = tatami::convert_to_compressed_sparse<int>(&raw_mat, true);

    auto rhs = tatami_test::simulate_sparse_vector<double>(NC * 2, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 421);
    tatami::DenseColumnMatrix<double, int> rhs_mat(NC, 2, rhs);
    auto rhs_spptr = tatami::convert_to_compressed_sparse(&rhs_mat, false);

    tatami::DenseRowMatrix<double, int> dmat(NR, NC, std::vector<double>(dump.begin(), dump.end()));
    std::vector<double> ref(NR * 2);
    tatami_mult::internal::sparse_row_tatami_dense(dmat, rhs_mat, ref.data(), 1);

    std::vector<double> output(NR * 2);
    tatami_mult::internal::sparse_row_tatami_dense(*spptr, rhs_mat, output.data(), 1);
    EXPECT_EQ(output, ref);
}
