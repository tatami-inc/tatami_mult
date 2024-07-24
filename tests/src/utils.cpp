#include <gtest/gtest.h>

#include "tatami_mult/utils.hpp"

TEST(Utils, Specials) {
    EXPECT_TRUE(tatami_mult::internal::supports_special_values<double>());
    EXPECT_FALSE(tatami_mult::internal::supports_special_values<int>());

    EXPECT_TRUE(tatami_mult::internal::is_special(std::numeric_limits<double>::infinity()));
    EXPECT_TRUE(tatami_mult::internal::is_special(std::numeric_limits<double>::quiet_NaN()));
    EXPECT_FALSE(tatami_mult::internal::is_special(0));
    EXPECT_FALSE(tatami_mult::internal::is_special(0.0));

    std::vector<double> contents { 0.0, std::numeric_limits<double>::infinity(), 1.0, std::numeric_limits<double>::quiet_NaN() };
    std::vector<int> specials;
    tatami_mult::internal::fill_special_index<int>(contents.size(), contents.data(), specials);
    std::vector<int> expected { 1, 3 };
    EXPECT_EQ(specials, expected);
}

TEST(Utils, SparseMultiply) {
    std::vector<double> sp_values { 0.5, 1.5, 2.5 };
    std::vector<int> sp_indices { 1, 3, 5 };
    tatami::SparseRange<double, int> range;
    range.number = sp_values.size();
    range.value = sp_values.data();
    range.index = sp_indices.data();

    std::vector<double> dense_values { 0, 1, 2, 3, 4, 5, 6 };
    double expected = 0.5 * 1 + 1.5 * 3 + 2.5 * 5;
    EXPECT_EQ(tatami_mult::internal::dense_sparse_multiply<double>(dense_values.data(), range), expected);

    // Same behavior with no specials.
    {
        std::vector<int> specials;
        EXPECT_EQ(tatami_mult::internal::special_dense_sparse_multiply<double>(specials, dense_values.data(), range), expected);
    }

    // Now only specials:
    {
        std::vector<int> specials { 0, 1, 2, 3, 4, 5, 6 };
        std::vector<double> new_dense_values(7, std::numeric_limits<double>::infinity());
        tatami::SparseRange<double, int> empty;
        EXPECT_TRUE(std::isnan(tatami_mult::internal::special_dense_sparse_multiply<double>(specials, new_dense_values.data(), empty)));
    }

    // Getting an Inf back:
    {
        std::vector<double> copy = dense_values;
        for (auto i : sp_indices) {
            copy[i] = std::numeric_limits<double>::infinity();
        }
        EXPECT_TRUE(std::isinf(tatami_mult::internal::special_dense_sparse_multiply<double>(sp_indices, copy.data(), range)));
    }

    // Getting an NaN back:
    {
        std::vector<int> specials { 0, 2, 4, 6 };
        std::vector<double> copy = dense_values;
        for (auto i : specials) {
            copy[i] = std::numeric_limits<double>::infinity();
        }
        EXPECT_TRUE(std::isnan(tatami_mult::internal::special_dense_sparse_multiply<double>(specials, copy.data(), range)));
    }

    // Throwing in one NaN at each position and checking we get an NaN back.
    for (size_t i = 0; i < dense_values.size(); ++i) {
        auto copy = dense_values;
        std::vector<int> specials;
        specials.push_back(i);
        copy[i] = std::numeric_limits<double>::quiet_NaN();
        EXPECT_TRUE(std::isnan(tatami_mult::internal::special_dense_sparse_multiply<double>(specials, copy.data(), range)));
    }

    // Throwing in an +Inf and then a -Inf and checking we get an NaN back.
    // (This is also the case if we have Inf * 0, which gives us NaN.)
    for (size_t i = 1; i < dense_values.size(); ++i) {
        auto copy = dense_values;
        std::vector<int> specials;
        specials.push_back(i - 1);
        copy[i - 1] = -std::numeric_limits<double>::infinity();
        specials.push_back(i);
        copy[i] = std::numeric_limits<double>::infinity();
        EXPECT_TRUE(std::isnan(tatami_mult::internal::special_dense_sparse_multiply<double>(specials, copy.data(), range)));
    }
}
