#include <gtest/gtest.h>

#include "tatami_mult/dense_row.hpp"
#include "tatami_test/tatami_test.hpp"

#include "utils.h"

class DenseRowTest : public ::testing::Test {
public:
    inline static size_t NR, NC;
    inline static std::vector<double> dump;
    inline static std::shared_ptr<tatami::Matrix<double, int> > dense;

    static void SetUpTestSuite() {
        NR = 102;
        NC = 92;
        dump = tatami_test::simulate_dense_vector<double>(NR * NC, /* lower = */ -10, /* upper = */ 10, /* seed = */ 69);
        dense.reset(new tatami::DenseRowMatrix<double, int>(NR, NC, dump));
    }
};

TEST_F(DenseRowTest, Vector) {
    auto rhs = tatami_test::simulate_dense_vector<double>(NC, /* lower = */ -10, /* upper = */ 10, /* seed = */ 421);

    // Doing a reference calculation.
    std::vector<double> ref(NR);
    for (size_t r = 0; r < NR; ++r) {
        ref[r] = std::inner_product(rhs.begin(), rhs.end(), dump.begin() + r * NC, 0.0);
    }

    for (int thread = 1; thread < 4; thread +=2) {
        std::vector<double> output(NR);
        tatami_mult::internal::dense_row_vector(*dense, rhs.data(), output.data(), thread);
        EXPECT_EQ(output, ref);
    }
}

TEST_F(DenseRowTest, Vectors) {
    auto raw_rhs = tatami_test::simulate_dense_vector<double>(NC * 2, /* lower = */ -10, /* upper = */ 10, /* seed = */ 422);
    std::vector<double*> rhs{ raw_rhs.data(), raw_rhs.data() + NC };

    // Doing a reference calculation.
    std::vector<double> ref(NR * 2);
    tatami_mult::internal::dense_row_vector(*dense, rhs[0], ref.data(), 1);
    tatami_mult::internal::dense_row_vector(*dense, rhs[1], ref.data() + NR, 1);

    for (int thread = 1; thread < 4; thread +=2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::dense_row_vectors(*dense, rhs, output.data(), thread);
        EXPECT_EQ(output, ref);
    }
}

TEST_F(DenseRowTest, TatamiDense) {
    auto raw_rhs = tatami_test::simulate_dense_vector<double>(NC * 2, /* lower = */ -10, /* upper = */ 10, /* seed = */ 423);
    std::vector<double*> rhs_ptrs { raw_rhs.data(), raw_rhs.data() + NC };
    auto rhs_dense = std::make_shared<tatami::DenseColumnMatrix<double, int> >(NC, 2, raw_rhs);

    // Doing a reference calculation.
    std::vector<double> ref(NR * 2);
    tatami_mult::internal::dense_row_vectors(*dense, rhs_ptrs, ref.data(), 1);

    for (int thread = 1; thread < 4; thread += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::dense_row_tatami_dense(*dense, *rhs_dense, output.data(), thread);
        EXPECT_EQ(output, ref);
    }
}

TEST_F(DenseRowTest, TatamiSparse) {
    auto rhs = tatami_test::simulate_sparse_vector<double>(NC * 2, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 424);
    auto rhs_dense = std::make_shared<tatami::DenseColumnMatrix<double, int> >(NC, 2, std::move(rhs));
    auto rhs_sparse = tatami::convert_to_compressed_sparse(rhs_dense.get(), false);

    // Doing a reference calculation.
    std::vector<double> ref(NR * 2);
    tatami_mult::internal::dense_row_tatami_dense(*dense, *rhs_dense, ref.data(), 1);

    for (int thread = 1; thread < 4; thread += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::dense_row_tatami_sparse(*dense, *rhs_sparse, output.data(), thread);
        EXPECT_EQ(output, ref);
    }
}

TEST_F(DenseRowTest, TatamiSparseSpecial) {
    auto dump2 = dump;
    for (size_t r = 0; r < NR; ++r) {
        dump2[r * NC] = std::numeric_limits<double>::infinity();
    }
    auto dense2 = std::make_shared<tatami::DenseRowMatrix<double, int> >(NR, NC, dump2);

    auto rhs = tatami_test::simulate_sparse_vector<double>(NC * 2, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 426);
    rhs[0] = 10; // making sure that we get something non-zero multiplying an Inf, which gives us another Inf.
    rhs[NC] = 0; // now trying with a zero multiplying an Inf, which gives us NaN.
    auto rhs_dense = std::make_shared<tatami::DenseColumnMatrix<double, int> >(NC, 2, std::move(rhs));
    auto rhs_sparse = tatami::convert_to_compressed_sparse(rhs_dense.get(), false);

    // Doing a reference calculation.
    std::vector<double> ref(NR * 2);
    tatami_mult::internal::dense_row_tatami_dense(*dense2, *rhs_dense, ref.data(), 1);

    for (int thread = 1; thread < 4; thread += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::dense_row_tatami_sparse(*dense2, *rhs_sparse, output.data(), thread);
        expect_equal_with_nan(ref, output);
    }
}

TEST_F(DenseRowTest, TatamiSparseNoSpecial) {
    auto idump = dump;
    for (auto& x : idump) {
        x = std::round(x);
    }
    auto idense = std::make_shared<tatami::DenseRowMatrix<double, int> >(NR, NC, idump);
    auto idense2 = tatami::convert_to_dense<int>(idense.get(), true); 

    auto rhs = tatami_test::simulate_sparse_vector<double>(NC * 2, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 421);
    auto rhs_dense = std::make_shared<tatami::DenseColumnMatrix<double, int> >(NC, 2, std::move(rhs));
    auto rhs_sparse = tatami::convert_to_compressed_sparse(rhs_dense.get(), false);

    std::vector<double> ref(NR * 2);
    tatami_mult::internal::dense_row_tatami_sparse(*idense, *rhs_sparse, ref.data(), 1);

    for (int thread = 1; thread < 4; thread += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::dense_row_tatami_sparse(*idense2, *rhs_sparse, output.data(), thread);
        EXPECT_EQ(ref, output);
    }
}
