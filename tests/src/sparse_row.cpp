#include <gtest/gtest.h>

#include "tatami_mult/sparse_row.hpp"
#include "tatami_test/tatami_test.hpp"

#include "utils.h"

class SparseRowTest : public ::testing::Test {
protected:
    inline static size_t NR, NC;
    inline static std::vector<double> dump;
    inline static std::shared_ptr<tatami::Matrix<double, int> > sparse;

    static void SetUpTestSuite() {
        NR = 99;
        NC = 152;
        dump = tatami_test::simulate_sparse_vector<double>(NR * NC, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 99);
        tatami::DenseRowMatrix<double, int> dense(NR, NC, dump);
        sparse = tatami::convert_to_compressed_sparse(&dense, true);
    }

    static tatami::DenseRowMatrix<double, int> rounded() {
        auto idump = dump;
        for (auto& x : idump) {
            x = std::round(x);
        }
        return tatami::DenseRowMatrix<double, int>(NR, NC, std::move(idump));
    }
};

TEST_F(SparseRowTest, Vector) {
    auto rhs = tatami_test::simulate_sparse_vector<double>(NC, /* lower = */ -10, /* upper = */ 10, /* seed = */ 421);

    // Doing a reference calculation.
    std::vector<double> ref(NR);
    for (size_t r = 0; r < NR; ++r) {
        ref[r] = std::inner_product(rhs.begin(), rhs.end(), dump.begin() + r * NC, 0.0);
    }

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR);
        tatami_mult::internal::sparse_row_vector(*sparse, rhs.data(), output.data(), threads);
        EXPECT_EQ(output, ref);
    }
}

TEST_F(SparseRowTest, VectorSpecial) {
    auto rhs = tatami_test::simulate_sparse_vector<double>(NC, /* lower = */ -10, /* upper = */ 10, /* seed = */ 422);
    rhs[0] = std::numeric_limits<double>::quiet_NaN();

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR);
        tatami_mult::internal::sparse_row_vector(*sparse, rhs.data(), output.data(), threads);
        for (auto x : output) {
            EXPECT_TRUE(std::isnan(x));
        }
    }
}

TEST_F(SparseRowTest, VectorNoSpecial) {
    auto dmat = rounded();
    auto spptr1 = tatami::convert_to_compressed_sparse(&dmat, true);
    auto spptr2 = tatami::convert_to_compressed_sparse<int>(&dmat, true);

    auto rhs = tatami_test::simulate_sparse_vector<double>(NC, /* lower = */ -10, /* upper = */ 10, /* seed = */ 423);

    std::vector<double> ref(NR);
    tatami_mult::internal::sparse_row_vector(*spptr1, rhs.data(), ref.data(), 1);

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR);
        tatami_mult::internal::sparse_row_vector(*spptr2, rhs.data(), output.data(), threads);
        EXPECT_EQ(output, ref);
    }
}

TEST_F(SparseRowTest, Matrix) {
    auto rhs = tatami_test::simulate_dense_vector<double>(NC * 2, /* lower = */ -10, /* upper = */ 10, /* seed = */ 424);

    // Doing a reference calculation.
    std::vector<double> ref(NR * 2);
    tatami_mult::internal::sparse_row_vector(*sparse, rhs.data(), ref.data(), 1);
    tatami_mult::internal::sparse_row_vector(*sparse, rhs.data() + NC, ref.data() + NR, 1);

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::sparse_row_matrix(*sparse, rhs.data(), 2, output.data(), threads);
        EXPECT_EQ(output, ref);
    }
}

TEST_F(SparseRowTest, MatrixSpecial) {
    auto rhs = tatami_test::simulate_dense_vector<double>(NC * 2, /* lower = */ -10, /* upper = */ 10, /* seed = */ 425);
    rhs[0] = std::numeric_limits<double>::quiet_NaN();
    rhs[2 * NC - 1] = std::numeric_limits<double>::infinity();

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::sparse_row_matrix(*sparse, rhs.data(), 2, output.data(), threads);
        for (size_t r = 0; r < NR; ++r) {
            EXPECT_TRUE(std::isnan(output[r]));
        }
        for (size_t r = 0; r < NR; ++r) {
            EXPECT_FALSE(std::isfinite(output[r + NR]));
        }
    }
}

TEST_F(SparseRowTest, MatrixNoSpecial) {
    auto dmat = rounded();
    auto spptr1 = tatami::convert_to_compressed_sparse(&dmat, true);
    auto spptr2 = tatami::convert_to_compressed_sparse<int>(&dmat, true);

    auto rhs = tatami_test::simulate_dense_vector<double>(NC * 2, /* lower = */ -10, /* upper = */ 10, /* seed = */ 426);

    std::vector<double> ref(NR * 2);
    tatami_mult::internal::sparse_row_matrix(*spptr1, rhs.data(), 2, ref.data(), 1);

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::sparse_row_matrix(*spptr2, rhs.data(), 2, output.data(), threads);
        EXPECT_EQ(output, ref);
    }
}

TEST_F(SparseRowTest, TatamiDense) {
    auto rhs = tatami_test::simulate_dense_vector<double>(NC * 2, /* lower = */ -10, /* upper = */ 10, /* seed = */ 427);
    tatami::DenseColumnMatrix<double, int> rhs_mat(NC, 2, rhs);

    std::vector<double> ref(NR * 2);
    tatami_mult::internal::sparse_row_matrix(*sparse, rhs.data(), 2, ref.data(), 1);

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::sparse_row_tatami_dense(*sparse, rhs_mat, output.data(), threads);
        EXPECT_EQ(output, ref);
    }
}

TEST_F(SparseRowTest, TatamiDenseSpecial) {
    auto rhs = tatami_test::simulate_dense_vector<double>(NC * 2, /* lower = */ -10, /* upper = */ 10, /* seed = */ 428);
    rhs[0] = std::numeric_limits<double>::quiet_NaN();
    rhs[2 * NC - 1] = std::numeric_limits<double>::infinity();
    tatami::DenseColumnMatrix<double, int> rhs_mat(NC, 2, rhs);

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::sparse_row_matrix(*sparse, rhs.data(), 2, output.data(), threads);
        for (size_t r = 0; r < NR; ++r) {
            EXPECT_TRUE(std::isnan(output[r]));
        }
        for (size_t r = 0; r < NR; ++r) {
            EXPECT_FALSE(std::isfinite(output[r + NR]));
        }
    }
}

TEST_F(SparseRowTest, TatamiDenseNoSpecial) {
    auto dmat = rounded();
    auto spptr1 = tatami::convert_to_compressed_sparse(&dmat, true);
    auto spptr2 = tatami::convert_to_compressed_sparse<int>(&dmat, true);

    auto rhs = tatami_test::simulate_dense_vector<double>(NC * 2, /* lower = */ -10, /* upper = */ 10, /* seed = */ 429);
    tatami::DenseColumnMatrix<double, int> rhs_mat(NC, 2, rhs);

    std::vector<double> ref(NR * 2);
    tatami_mult::internal::sparse_row_tatami_dense(*spptr1, rhs_mat, ref.data(), 1);

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::sparse_row_tatami_dense(*spptr2, rhs_mat, output.data(), threads);
        EXPECT_EQ(output, ref);
    }
}

TEST_F(SparseRowTest, TatamiSparse) {
    auto rhs = tatami_test::simulate_sparse_vector<double>(NC * 2, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 421);
    tatami::DenseColumnMatrix<double, int> rhs_mat(NC, 2, rhs);
    auto rhs_spptr = tatami::convert_to_compressed_sparse(&rhs_mat, false);

    // Doing a reference calculation.
    std::vector<double> ref(NR * 2);
    tatami_mult::internal::sparse_row_tatami_dense(*sparse, rhs_mat, ref.data(), 1);

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::sparse_row_tatami_sparse(*sparse, *rhs_spptr, output.data(), threads);
        EXPECT_EQ(output, ref);
    }
}

TEST_F(SparseRowTest, TatamiSparseSpecial) {
    auto rhs = tatami_test::simulate_sparse_vector<double>(NC * 2, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 421);
    rhs[0] = std::numeric_limits<double>::quiet_NaN();
    rhs[2 * NC - 1] = std::numeric_limits<double>::infinity();
    tatami::DenseColumnMatrix<double, int> rhs_mat(NC, 2, rhs);
    auto rhs_spptr = tatami::convert_to_compressed_sparse(&rhs_mat, false);

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::sparse_row_tatami_sparse(*sparse, *rhs_spptr, output.data(), threads);
        for (size_t r = 0; r < NR; ++r) {
            EXPECT_TRUE(std::isnan(output[r]));
        }
        for (size_t r = 0; r < NR; ++r) {
            EXPECT_FALSE(std::isfinite(output[r + NR]));
        }
    }
}

TEST_F(SparseRowTest, TatamiSparseNoSpecial) {
    auto dmat = rounded();
    auto spptr1 = tatami::convert_to_compressed_sparse(&dmat, true);
    auto spptr2 = tatami::convert_to_compressed_sparse<int>(&dmat, true);

    auto rhs = tatami_test::simulate_sparse_vector<double>(NC * 2, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 421);
    tatami::DenseColumnMatrix<double, int> rhs_mat(NC, 2, rhs);
    auto rhs_spptr = tatami::convert_to_compressed_sparse(&rhs_mat, false);

    std::vector<double> ref(NR * 2);
    tatami_mult::internal::sparse_row_tatami_dense(*spptr1, *rhs_spptr, ref.data(), 1);

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::sparse_row_tatami_dense(*spptr2, *rhs_spptr, output.data(), threads);
        EXPECT_EQ(output, ref);
    }
}
