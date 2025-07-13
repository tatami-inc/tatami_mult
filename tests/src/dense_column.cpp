#include <gtest/gtest.h>

#include "tatami_mult/dense_column.hpp"
#include "tatami_test/tatami_test.hpp"

#include "utils.h"

class DenseColumnTest : public ::testing::Test {
public:
    inline static int NR, NC;
    inline static std::vector<double> dump;
    inline static std::shared_ptr<tatami::Matrix<double, int> > dense;

    static void SetUpTestSuite() {
        NR = 61;
        NC = 192;

        dump = tatami_test::simulate_vector<double>(NR * NC, []{
            tatami_test::SimulateVectorOptions opt;
            opt.lower = -10;
            opt.upper = 10;
            opt.seed = 42;
            return opt;
        }());

        dense.reset(new tatami::DenseRowMatrix<double, int>(NR, NC, dump));
    }
};

TEST_F(DenseColumnTest, Vector) {
    auto rhs = tatami_test::simulate_vector<double>(NC, []{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 690;
        return opt;
    }());

    // Doing a reference calculation.
    std::vector<double> ref(NR);
    for (int r = 0; r < NR; ++r) {
        ref[r] = std::inner_product(rhs.begin(), rhs.end(), dump.begin() + r * NC, 0.0);
    }

    for (int thread = 1; thread < 4; thread +=2) {
        std::vector<double> output(NR);
        tatami_mult::internal::dense_column_vector(*dense, rhs.data(), output.data(), thread);
        EXPECT_EQ(output, ref);
    }
}

TEST_F(DenseColumnTest, Vectors) {
    auto raw_rhs = tatami_test::simulate_vector<double>(NC * 2, []{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 422;
        return opt;
    }());

    std::vector<double*> rhs{ raw_rhs.data(), raw_rhs.data() + NC };

    // Doing a reference calculation.
    std::vector<double> ref(NR * 2);
    tatami_mult::internal::dense_column_vector(*dense, rhs[0], ref.data(), 1);
    tatami_mult::internal::dense_column_vector(*dense, rhs[1], ref.data() + NR, 1);

    for (int thread = 1; thread < 4; thread +=2) {
        std::vector<double> output(NR * 2);
        std::vector<double*> out_ptrs{ output.data(), output.data() + NR };
        tatami_mult::internal::dense_column_vectors(*dense, rhs, out_ptrs, thread);
        EXPECT_EQ(output, ref);
    }
}

TEST_F(DenseColumnTest, TatamiDense) {
    auto raw_rhs = tatami_test::simulate_vector<double>(NC * 2, []{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = -10;
        opt.upper = 10;
        opt.seed = 423;
        return opt;
    }());

    std::vector<double*> rhs_ptrs { raw_rhs.data(), raw_rhs.data() + NC };
    auto rhs_dense = std::make_shared<tatami::DenseColumnMatrix<double, int> >(NC, 2, raw_rhs);

    // Doing a reference calculation.
    std::vector<double> ref(NR * 2);
    std::vector<double*> ref_ptrs{ ref.data(), ref.data() + NR };
    tatami_mult::internal::dense_column_vectors(*dense, rhs_ptrs, ref_ptrs, 1);

    for (int thread = 1; thread < 4; thread += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::dense_column_tatami_dense(*dense, *rhs_dense, output.data(), 1, NR, thread);
        EXPECT_EQ(output, ref);

        std::vector<double> toutput(NR * 2);
        tatami_mult::internal::dense_column_tatami_dense(*dense, *rhs_dense, toutput.data(), 2, 1, thread);
        std::fill(output.begin(), output.end(), 0);
        tatami::transpose(toutput.data(), NR, 2, output.data());
        EXPECT_EQ(output, ref);
    }
}

TEST_F(DenseColumnTest, TatamiSparse) {
    std::shared_ptr<tatami::Matrix<double, int> > rhs_dense, rhs_sparse;
    {
        auto rhs = tatami_test::simulate_vector<double>(NC * 2, []{
            tatami_test::SimulateVectorOptions opt;
            opt.density = 0.1;
            opt.lower = -10;
            opt.upper = 10;
            opt.seed = 424;
            return opt;
        }());
        rhs_dense = std::make_shared<tatami::DenseColumnMatrix<double, int> >(NC, 2, std::move(rhs));
        rhs_sparse = tatami::convert_to_compressed_sparse(rhs_dense.get(), false);
    }

    // Doing a reference calculation.
    std::vector<double> ref(NR * 2);
    tatami_mult::internal::dense_column_tatami_dense(*dense, *rhs_dense, ref.data(), 1, NR, 1);

    for (int thread = 1; thread < 4; thread += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::dense_column_tatami_sparse(*dense, *rhs_sparse, output.data(), 1, NR, thread);
        EXPECT_EQ(output, ref);

        std::vector<double> toutput(NR * 2);
        tatami_mult::internal::dense_column_tatami_sparse(*dense, *rhs_sparse, toutput.data(), 2, 1, thread);
        std::fill(output.begin(), output.end(), 0);
        tatami::transpose(toutput.data(), NR, 2, output.data());
        EXPECT_EQ(output, ref);
    }
}

TEST_F(DenseColumnTest, TatamiSparseSpecial) {
    // Injecting special values into the LHS, which gets checked
    // before attempting multiplication by the sparse RHS.
    std::shared_ptr<tatami::Matrix<double, int> > dense2;
    {
        auto dump2 = dump;
        for (int r = 0; r < NR; ++r) { 
            int scenario = r % 3;
            if (scenario == 0 || scenario == 2) { // adding Inf to the start of each row.
                dump2[r * NC] = std::numeric_limits<double>::infinity();
            }
            if (scenario == 1 || scenario == 2) { // adding -Inf to the end of each row.
                dump2[(r + 1) * NC - 1] = -std::numeric_limits<double>::infinity();
            }
        }
        dense2 = std::make_shared<tatami::DenseRowMatrix<double, int> >(NR, NC, std::move(dump2));
    }

    std::shared_ptr<tatami::Matrix<double, int> > rhs_dense, rhs_sparse;
    {
        auto rhs = tatami_test::simulate_vector<double>(NC * 6, []{
            tatami_test::SimulateVectorOptions opt;
            opt.density = 0.1;
            opt.lower = -10;
            opt.upper = 10;
            opt.seed = 426;
            return opt;
        }());

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

        rhs_dense = std::make_shared<tatami::DenseColumnMatrix<double, int> >(NC, 6, std::move(rhs));
        rhs_sparse = tatami::convert_to_compressed_sparse(rhs_dense.get(), false);
    }

    // Doing a reference calculation.
    std::vector<double> ref(NR * 6);
    tatami_mult::internal::dense_column_tatami_dense(*dense2, *rhs_dense, ref.data(), 1, NR, 1);

    for (int thread = 1; thread < 4; thread += 2) {
        std::vector<double> output(NR * 6);
        tatami_mult::internal::dense_column_tatami_sparse(*dense2, *rhs_sparse, output.data(), 1, NR, thread);
        expect_equal_with_nan(ref, output);
    }
}

TEST_F(DenseColumnTest, TatamiSparseNoSpecial) {
    // The special check is based on the type of the LHS, 
    // so we convert our LHS into an integer matrix.
    std::shared_ptr<tatami::Matrix<double, int> > idense;
    std::shared_ptr<tatami::Matrix<int, int> > idense2;
    {
        auto idump = dump;
        for (auto& x : idump) {
            x = std::round(x);
        }
        idense = std::make_shared<tatami::DenseRowMatrix<double, int> >(NR, NC, std::move(idump));
        idense2 = tatami::convert_to_dense<int>(idense.get(), true); 
    }

    std::shared_ptr<tatami::Matrix<double, int> > rhs_sparse;
    {
        auto rhs = tatami_test::simulate_vector<double>(NC * 2, []{
            tatami_test::SimulateVectorOptions opt;
            opt.density = 0.1;
            opt.lower = -10;
            opt.upper = 10;
            opt.seed = 421;
            return opt;
        }());

        auto rhs_dense = std::make_shared<tatami::DenseColumnMatrix<double, int> >(NC, 2, std::move(rhs));
        rhs_sparse = tatami::convert_to_compressed_sparse(rhs_dense.get(), false);
    }

    std::vector<double> ref(NR * 2);
    tatami_mult::internal::dense_column_tatami_sparse(*idense, *rhs_sparse, ref.data(), 1, NR, 1);

    for (int thread = 1; thread < 4; thread += 2) {
        std::vector<double> output(NR * 2);
        tatami_mult::internal::dense_column_tatami_sparse(*idense2, *rhs_sparse, output.data(), 1, NR, thread);
        EXPECT_EQ(ref, output);
    }
}
