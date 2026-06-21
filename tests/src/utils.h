#ifndef UTILS_H
#define UTILS_H

#include <gtest/gtest.h>

#include <cassert>

inline void expect_almost_equal(const std::vector<double>& ref, const std::vector<double>& observed) {
    const auto n = ref.size();
    ASSERT_EQ(n, observed.size());
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(ref[i], observed[i]) << " at position " << i << std::endl;
    }
}

inline std::vector<double*> populate_pointers(std::vector<double>& buffer, size_t shift, size_t count) {
    assert(buffer.size() == shift * count);
    std::vector<double*> output(count);
    size_t offset = 0;
    for (size_t o = 0; o < count; ++o, offset += shift) {
        output[o] = buffer.data() + offset;
    }
    return output;
}

#endif
