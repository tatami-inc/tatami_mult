#include <gtest/gtest.h>

#include "tatami_mult/sparse_column.hpp"
#include "tatami_test/tatami_test.hpp"

#include "utils.h"

class SparseColumnTest : public ::testing::Test {
protected:
    inline static size_t NR, NC;
    inline static std::vector<double> dump;
    inline static std::shared_ptr<tatami::Matrix<double, int> > sparse;

    static void SetUpTestSuite() {
        NR = 199;
        NC = 52;
        dump = tatami_test::simulate_sparse_vector<double>(NR * NC, 0.2, /* lower = */ -10, /* upper = */ 10, /* seed = */ 99);
        tatami::DenseRowMatrix<double, int> dense(NR, NC, dump);
        sparse = tatami::convert_to_compressed_sparse(&dense, true);
    }
};

TEST_F(SparseColumnTest, Vector) {
    auto rhs = tatami_test::simulate_sparse_vector<double>(NC, /* lower = */ -10, /* upper = */ 10, /* seed = */ 421);

    // Doing a reference calculation.
    std::vector<double> ref(NR);
    for (size_t r = 0; r < NR; ++r) {
        ref[r] = std::inner_product(rhs.begin(), rhs.end(), dump.begin() + r * NC, 0.0);
    }

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR);
        tatami_mult::internal::sparse_column_vector(*sparse, rhs.data(), output.data(), threads);
        EXPECT_EQ(output, ref);
    }
}

TEST_F(SparseColumnTest, VectorSpecial) {
    for (int scenario = 0; scenario < 3; ++scenario) {
        auto rhs = tatami_test::simulate_sparse_vector<double>(NC, /* lower = */ -10, /* upper = */ 10, /* seed = */ 422 + scenario);

        if (scenario == 0) {
            rhs[0] = std::numeric_limits<double>::infinity();
        } else if (scenario == 1) {
            rhs[NC - 1] = std::numeric_limits<double>::infinity();
        } else {
            rhs[0] = std::numeric_limits<double>::infinity();
            rhs[NC - 1] = -std::numeric_limits<double>::infinity();
        }

        // Doing a reference calculation.
        std::vector<double> ref(NR);
        for (size_t r = 0; r < NR; ++r) {
            ref[r] = std::inner_product(rhs.begin(), rhs.end(), dump.begin() + r * NC, 0.0);
        }

        for (int threads = 1; threads < 4; threads += 2) {
            std::vector<double> output(NR);
            tatami_mult::internal::sparse_column_vector(*sparse, rhs.data(), output.data(), threads);
            expect_equal_with_nan(ref, output);
        }
    }
}

TEST_F(SparseColumnTest, VectorNoSpecial) {
    auto rhs_i = tatami_test::simulate_sparse_vector<int>(NC, /* lower = */ -10, /* upper = */ 10, /* seed = */ 423);
    std::vector<double> rhs_d(rhs_i.begin(), rhs_i.end());

    std::vector<double> ref(NR);
    tatami_mult::internal::sparse_column_vector(*sparse, rhs_d.data(), ref.data(), 1);

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR);
        tatami_mult::internal::sparse_column_vector(*sparse, rhs_i.data(), output.data(), threads);
        EXPECT_EQ(output, ref);
    }
}

TEST_F(SparseColumnTest, Vectors) {
    auto raw_rhs = tatami_test::simulate_dense_vector<double>(NC * 2, /* lower = */ -10, /* upper = */ 10, /* seed = */ 424);
    std::vector<double*> rhs { raw_rhs.data(), raw_rhs.data() + NC };

    // Doing a reference calculation.
    std::vector<double> ref(NR * 2);
    tatami_mult::internal::sparse_column_vector(*sparse, rhs[0], ref.data(), 1);
    tatami_mult::internal::sparse_column_vector(*sparse, rhs[1], ref.data() + NR, 1);

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::sparse_column_vectors(*sparse, rhs, output.data(), threads);
        EXPECT_EQ(output, ref);
    }
}

TEST_F(SparseColumnTest, VectorsSpecial) {
    auto raw_rhs = tatami_test::simulate_dense_vector<double>(NC * 3, /* lower = */ -10, /* upper = */ 10, /* seed = */ 425);
    std::vector<double*> rhs { raw_rhs.data(), raw_rhs.data() + NC, raw_rhs.data() + 2 * NC };
    rhs[0][0] = std::numeric_limits<double>::infinity();
    rhs[1][NC - 1] = std::numeric_limits<double>::infinity();
    rhs[2][0] = std::numeric_limits<double>::infinity();
    rhs[2][NC - 1] = std::numeric_limits<double>::infinity();

    // Doing a reference calculation.
    std::vector<double> ref(NR * 3);
    tatami_mult::internal::sparse_column_vector(*sparse, rhs[0], ref.data(), 1);
    tatami_mult::internal::sparse_column_vector(*sparse, rhs[1], ref.data() + NR, 1);
    tatami_mult::internal::sparse_column_vector(*sparse, rhs[2], ref.data() + NR * 2, 1);

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR * 3);
        tatami_mult::internal::sparse_column_vectors(*sparse, rhs, output.data(), threads);
        expect_equal_with_nan(ref, output);
    }
}

TEST_F(SparseColumnTest, VectorsNoSpecial) {
    auto raw_rhs_i = tatami_test::simulate_dense_vector<int>(NC * 2, /* lower = */ -10, /* upper = */ 10, /* seed = */ 426);
    std::vector<int*> rhs_i { raw_rhs_i.data(), raw_rhs_i.data() + NC };
    std::vector<double> raw_rhs_d(raw_rhs_i.begin(), raw_rhs_i.end());
    std::vector<double*> rhs_d { raw_rhs_d.data(), raw_rhs_d.data() + NC };

    std::vector<double> ref(NR * 2);
    tatami_mult::internal::sparse_column_vectors(*sparse, rhs_d, ref.data(), 1);

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::sparse_column_vectors(*sparse, rhs_i, output.data(), threads);
        EXPECT_EQ(output, ref);
    }
}

TEST_F(SparseColumnTest, TatamiDense) {
    auto raw_rhs = tatami_test::simulate_dense_vector<double>(NC * 2, /* lower = */ -10, /* upper = */ 10, /* seed = */ 427);
    std::vector<double*> rhs_ptrs { raw_rhs.data(), raw_rhs.data() + NC };
    auto rhs_dense = std::make_shared<tatami::DenseColumnMatrix<double, int> >(NC, 2, raw_rhs);

    std::vector<double> ref(NR * 2);
    tatami_mult::internal::sparse_column_vectors(*sparse, rhs_ptrs, ref.data(), 1);

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::sparse_column_tatami_dense(*sparse, *rhs_dense, output.data(), threads);
        EXPECT_EQ(output, ref);
    }
}

TEST_F(SparseColumnTest, TatamiDenseSpecial) {
    auto raw_rhs = tatami_test::simulate_dense_vector<double>(NC * 3, /* lower = */ -10, /* upper = */ 10, /* seed = */ 428);
    raw_rhs[0] = std::numeric_limits<double>::infinity(); // start of first column.
    raw_rhs[2 * NC - 1] = std::numeric_limits<double>::infinity(); // end of second column.
    raw_rhs[2 * NC] = std::numeric_limits<double>::infinity(); // start of third column.
    raw_rhs[3 * NC - 1] = std::numeric_limits<double>::infinity(); // end of third column.
    auto rhs_dense = std::make_shared<tatami::DenseColumnMatrix<double, int> >(NC, 3, raw_rhs);

    std::vector<double> ref(NR * 3);
    std::vector<double*> rhs_ptrs { raw_rhs.data(), raw_rhs.data() + NC, raw_rhs.data() + 2 * NC };
    tatami_mult::internal::sparse_column_vectors(*sparse, rhs_ptrs, ref.data(), 1);

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR * 3);
        tatami_mult::internal::sparse_column_tatami_dense(*sparse, *rhs_dense, output.data(), threads);
        expect_equal_with_nan(ref, output);
    }
}

TEST_F(SparseColumnTest, TatamiDenseNoSpecial) {
    auto raw_rhs_i = tatami_test::simulate_dense_vector<int>(NC * 2, /* lower = */ -10, /* upper = */ 10, /* seed = */ 429);
    auto rhs_i = std::make_shared<tatami::DenseColumnMatrix<int, int> >(NC, 2, raw_rhs_i);
    auto rhs_d = std::make_shared<tatami::DenseColumnMatrix<double, int> >(NC, 2, std::vector<double>(raw_rhs_i.begin(), raw_rhs_i.end()));

    std::vector<double> ref(NR * 2);
    tatami_mult::internal::sparse_column_tatami_dense(*sparse, *rhs_d, ref.data(), 1);

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::sparse_column_tatami_dense(*sparse, *rhs_i, output.data(), threads);
        EXPECT_EQ(output, ref);
    }
}

TEST_F(SparseColumnTest, TatamiSparse) {
    std::shared_ptr<tatami::Matrix<double, int> > rhs_dense, rhs_sparse;
    {
        auto rhs = tatami_test::simulate_sparse_vector<double>(NC * 2, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 430);
        rhs_dense = std::make_shared<tatami::DenseColumnMatrix<double, int> >(NC, 2, std::move(rhs));
        rhs_sparse = tatami::convert_to_compressed_sparse(rhs_dense.get(), false);
    }


    // Doing a reference calculation.
    std::vector<double> ref(NR * 2);
    tatami_mult::internal::sparse_column_tatami_dense(*sparse, *rhs_dense, ref.data(), 1);

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::sparse_column_tatami_sparse(*sparse, *rhs_sparse, output.data(), threads);
        EXPECT_EQ(output, ref);
    }
}

TEST_F(SparseColumnTest, TatamiSparseSpecial) {
    std::shared_ptr<tatami::Matrix<double, int> > sparse2;
    {
        auto dump2 = dump;
        for (size_t r = 0; r < NR; ++r) { 
            int scenario = r % 3;
            if (scenario == 0 || scenario == 2) { // adding Inf to the start of each row.
                dump2[r * NC] = std::numeric_limits<double>::infinity();
            }
            if (scenario == 1 || scenario == 2) { // adding -Inf to the end of each row.
                dump2[(r + 1) * NC - 1] = -std::numeric_limits<double>::infinity();
            }
        }
        auto dense2 = std::make_shared<tatami::DenseRowMatrix<double, int> >(NR, NC, std::move(dump2));
        sparse2 = tatami::convert_to_compressed_sparse(dense2.get(), false);
    }

    std::shared_ptr<tatami::Matrix<double, int> > rhs_dense, rhs_sparse;
    {
        auto rhs = tatami_test::simulate_sparse_vector<double>(NC * 6, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 431);

        // Remember, we're injecting Inf's to the first and/or last values of each
        // row of the LHS matrix. So we need to set some of the RHS values to
        // ensure that we get a good mix of NaNs and Infs in the output; otherwise
        // we couldn't tell if the specials were processed correctly.
        // Specifically, we set the first/last values of each RHS column to 0
        // and/or 10, which yield NaN and Inf respectively when multiplied by Inf.
        rhs[0] = 10;          // (0, 0)
        rhs[NC] = 0;          // (0, 1)
        rhs[4 * NC] = 0;      // (0, 4)
        rhs[5 * NC] = 0;      // (0, 5)
        rhs[3 * NC - 1] = 10; // (NC-1, 2)
        rhs[4 * NC - 1] = 0;  // (NC-1, 3)
        rhs[5 * NC - 1] = 10; // (NC-1, 4)
        rhs[6 * NC - 1] = 0;  // (NC-1, 5)

        rhs_dense = std::make_shared<tatami::DenseColumnMatrix<double, int> >(NC, 6, rhs);
        rhs_sparse = tatami::convert_to_compressed_sparse(rhs_dense.get(), false);
    }

    // Doing a reference calculation.
    std::vector<double> ref(NR * 6);
    tatami_mult::internal::sparse_column_tatami_dense(*sparse2, *rhs_dense, ref.data(), 1);

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR * 6);
        tatami_mult::internal::sparse_column_tatami_sparse(*sparse2, *rhs_sparse, output.data(), threads);
        expect_equal_with_nan(ref, output);
    }
}

TEST_F(SparseColumnTest, TatamiSparseNoSpecial) {
    // The special check is based on the type of the LHS this time,
    // so we convert our LHS into an integer matrix.
    std::shared_ptr<tatami::Matrix<double, int> > sparse_d;
    std::shared_ptr<tatami::Matrix<int, int> > sparse_i;
    {
        auto dump_i = dump;
        for (auto& x : dump_i) {
            x = std::round(x);
        }
        auto dense_d = std::make_shared<tatami::DenseRowMatrix<double, int> >(NR, NC, std::move(dump_i));
        sparse_d = tatami::convert_to_dense<double>(dense_d.get(), true);
        sparse_i = tatami::convert_to_dense<int>(dense_d.get(), true);
    }

    std::shared_ptr<tatami::Matrix<double, int> > rhs_sparse;
    {
        auto raw_rhs = tatami_test::simulate_sparse_vector<double>(NC * 2, 0.1, /* lower = */ -10, /* upper = */ 10, /* seed = */ 432);
        auto rhs_dense = std::make_shared<tatami::DenseColumnMatrix<double, int> >(NC, 2, std::move(raw_rhs));
        rhs_sparse = tatami::convert_to_compressed_sparse<double>(rhs_dense.get(), false);
    }

    std::vector<double> ref(NR * 2);
    tatami_mult::internal::sparse_column_tatami_sparse(*sparse_d, *rhs_sparse, ref.data(), 1);

    for (int threads = 1; threads < 4; threads += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::sparse_column_tatami_sparse(*sparse_i, *rhs_sparse, output.data(), threads);
        EXPECT_EQ(output, ref);
    }
}
