#ifndef UTILS_H
#define UTILS_H

#include <gtest/gtest.h>

inline void expect_equal_with_nan(const std::vector<double>& ref, const std::vector<double>& observed) {
    ASSERT_EQ(ref.size(), observed.size());
    size_t n = ref.size();
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(std::isnan(ref[i]), std::isnan(observed[i])) << " at position " << i << std::endl;
        if (!std::isnan(ref[i])) {
            EXPECT_EQ(ref[i], observed[i]) << " at position " << i << std::endl;
        }
    }
}

#endif
