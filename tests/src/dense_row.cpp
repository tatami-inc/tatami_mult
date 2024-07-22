#include <gtest/gtest.h>

#include "tatami_mult/dense_row.hpp"
#include "tatami_test/tatami_test.hpp"

#include "utils.h"

TEST(DenseRow, Vector) {
    size_t NR = 99, NC = 152;
    auto dump = tatami_test::simulate_dense_vector<double>(NR * NC);
    tatami::DenseRowMatrix<double, int> mat(NR, NC, dump);
    auto rhs = tatami_test::simulate_dense_vector<double>(NC, /* lower = */ -10, /* upper = */ 10, /* seed = */ 421);

    // Doing a reference calculation.
    std::vector<double> ref(NR);
    for (size_t r = 0; r < NR; ++r) {
        ref[r] = std::inner_product(rhs.begin(), rhs.end(), dump.begin() + r * NC, 0.0);
    }

    std::vector<double> output(NR);
    tatami_mult::internal::dense_row_vector(mat, rhs.data(), output.data(), 1);
    EXPECT_EQ(output, ref);

    // Same results with multiple threads.
    std::fill(output.begin(), output.end(), 0);
    tatami_mult::internal::dense_row_vector(mat, rhs.data(), output.data(), 3);
    EXPECT_EQ(output, ref);
}

TEST(DenseRow, Matrix) {
    size_t NR = 129, NC = 60;
    auto dump = tatami_test::simulate_dense_vector<double>(NR * NC);
    tatami::DenseRowMatrix<double, int> mat(NR, NC, std::move(dump));

    auto rhs = tatami_test::simulate_dense_vector<double>(NC * 2, /* lower = */ -10, /* upper = */ 10, /* seed = */ 421);

    // Doing a reference calculation.
    std::vector<double> ref(NR * 2);
    tatami_mult::internal::dense_row_vector(mat, rhs.data(), ref.data(), 1);
    tatami_mult::internal::dense_row_vector(mat, rhs.data() + NC, ref.data() + NR, 1);

    std::vector<double> output(NR * 2);
    tatami_mult::internal::dense_row_matrix(mat, rhs.data(), 2, output.data(), 1);
    EXPECT_EQ(output, ref);

    // Same results with multiple threads.
    std::fill(output.begin(), output.end(), 0);
    tatami_mult::internal::dense_row_matrix(mat, rhs.data(), 2, output.data(), 3);
    EXPECT_EQ(output, ref);
}

TEST(DenseRow, TatamiDense) {
    size_t NR = 131, NC = 51;
    auto dump = tatami_test::simulate_dense_vector<double>(NR * NC);
    tatami::DenseRowMatrix<double, int> mat(NR, NC, dump);

    auto rhs = tatami_test::simulate_dense_vector<double>(NC * 2, /* lower = */ -10, /* upper = */ 10, /* seed = */ 421);
    tatami::DenseColumnMatrix<double, int> rhs_mat(NC, 2, rhs);

    // Doing a reference calculation.
    std::vector<double> ref(NR * 2);
    tatami_mult::internal::dense_row_matrix(mat, rhs.data(), 2, ref.data(), 1);

    std::vector<double> output(NR * 2);
    tatami_mult::internal::dense_row_tatami_dense(mat, rhs_mat, output.data(), 1);
    EXPECT_EQ(output, ref);

    // Same results with multiple threads.
    std::fill(output.begin(), output.end(), 0);
    tatami_mult::internal::dense_row_tatami_dense(mat, rhs_mat, output.data(), 3);
    EXPECT_EQ(output, ref);
}

TEST(DenseRow, TatamiSparse) {
    size_t NR = 131, NC = 51;
    auto dump = tatami_test::simulate_dense_vector<double>(NR * NC);
    tatami::DenseRowMatrix<double, int> mat(NR, NC, dump);

    auto rhs = tatami_test::simulate_sparse_vector<double>(NC * 2, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 421);
    tatami::DenseColumnMatrix<double, int> rhs_mat(NC, 2, std::move(rhs));
    auto rhs_spmat = tatami::convert_to_compressed_sparse(&rhs_mat, false);

    // Doing a reference calculation.
    std::vector<double> ref(NR * 2);
    tatami_mult::internal::dense_row_tatami_dense(mat, rhs_mat, ref.data(), 1);

    std::vector<double> output(NR * 2);
    tatami_mult::internal::dense_row_tatami_sparse(mat, *rhs_spmat, output.data(), 1);
    EXPECT_EQ(output, ref);

    // Same results with multiple threads.
    std::fill(output.begin(), output.end(), 0);
    tatami_mult::internal::dense_row_tatami_sparse(mat, *rhs_spmat, output.data(), 3);
    EXPECT_EQ(output, ref);
}

TEST(DenseRow, TatamiSparseSpecial) {
    size_t NR = 131, NC = 51;
    auto dump = tatami_test::simulate_dense_vector<double>(NR * NC);
    for (size_t r = 0; r < NR; ++r) {
        dump[r * NC] = std::numeric_limits<double>::infinity();
    }
    tatami::DenseRowMatrix<double, int> mat(NR, NC, dump);

    auto rhs = tatami_test::simulate_sparse_vector<double>(NC * 2, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 421);
    rhs[0] = 10; // making sure that we get something non-zero multiplying an Inf, which gives us another Inf.
    rhs[NC] = 0; // now trying with a zero multiplying an Inf, which gives us NaN.
    tatami::DenseColumnMatrix<double, int> rhs_mat(NC, 2, std::move(rhs));
    auto rhs_spmat = tatami::convert_to_compressed_sparse(&rhs_mat, false);

    // Doing a reference calculation.
    std::vector<double> ref(NR * 2);
    tatami_mult::internal::dense_row_tatami_dense(mat, rhs_mat, ref.data(), 1);

    std::vector<double> output(NR * 2);
    tatami_mult::internal::dense_row_tatami_sparse(mat, *rhs_spmat, output.data(), 1);
    expect_equal_with_nan(ref, output);

    // Same results with multiple threads.
    std::fill(output.begin(), output.end(), 0);
    tatami_mult::internal::dense_row_tatami_sparse(mat, *rhs_spmat, output.data(), 3);
    expect_equal_with_nan(ref, output);
}

TEST(DenseRow, TatamiSparseNoSpecial) {
    // NaNs are impossible with integer matrices.
    size_t NR = 89, NC = 151;
    auto dump = tatami_test::simulate_dense_vector<int>(NR * NC, /* lower = */ -10, /* upper = */ 10, /* seed = */ 69);
    tatami::DenseRowMatrix<int, int> mat(NR, NC, dump);

    auto rhs = tatami_test::simulate_sparse_vector<double>(NC * 2, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 421);
    tatami::DenseColumnMatrix<double, int> rhs_mat(NC, 2, std::move(rhs));
    auto rhs_spmat = tatami::convert_to_compressed_sparse(&rhs_mat, false);

    std::vector<double> ref(NR * 2);
    tatami_mult::internal::dense_row_tatami_dense(mat, rhs_mat, ref.data(), 1);

    std::vector<double> output(NR * 2);
    tatami_mult::internal::dense_row_tatami_sparse(mat, *rhs_spmat, output.data(), 1);
    EXPECT_EQ(ref, output);
}
