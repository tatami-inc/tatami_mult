#ifndef TATAMI_MULT_UTILS_HPP
#define TATAMI_MULT_UTILS_HPP

#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"

namespace tatami_mult {

namespace internal {

template<typename Value_>
constexpr bool supports_special_values() {
    return (std::numeric_limits<Value_>::has_infinity || std::numeric_limits<Value_>::has_quiet_NaN || std::numeric_limits<Value_>::has_signaling_NaN);
}

template<typename Value_>
bool is_special(Value_ x) {
    return !std::isfinite(x);
}

template<typename Index_, typename Value_>
void fill_special_index(Index_ len, const Value_* ptr, std::vector<Index_>& specials) {
    for (Index_ i = 0; i < len; ++i) {
        if (is_special(ptr[i])) {
            specials.push_back(i);
        }
    }
}

template<typename Output_, typename DenseValue_, typename Value_, typename Index_>
Output_ dense_sparse_multiply(const DenseValue_* ptr, const tatami::SparseRange<Value_, Index_>& range) {
    Output_ out = 0;
    for (Index_ k = 0; k < range.number; ++k) {
        out += range.value[k] * ptr[range.index[k]];
    }
    return out;
}

template<typename Output_, typename SpecialIndex_, typename SpecialValue_, typename Value_, typename Index_>
Output_ special_dense_sparse_multiply(const std::vector<SpecialIndex_>& specials, const SpecialValue_* ptr, const tatami::SparseRange<Value_, Index_>& range) {
    Output_ out = 0;
    auto sIt = specials.begin(), sEnd = specials.end();
    Index_ k = 0;

    if (k < range.number && sIt != sEnd) {
        Index_ spec = *sIt;
        Index_ ridx = range.index[k];
        while (1) {
            if (ridx < spec) {
                out += ptr[ridx] * range.value[k]; // need to multiply by zero in case the range.value is special.
                if (++k == range.number) {
                    break;
                }
                ridx = range.index[k];
            } else if (ridx > spec) {
                out += ptr[spec] * static_cast<Value_>(0); // it's special, so we can't assume the product would be zero.
                if (++sIt == sEnd) {
                    break;
                }
                spec = *sIt;
            } else {
                out += ptr[spec] * range.value[k];
                ++k;
                ++sIt;
                if (k == range.number || sIt == sEnd) {
                    break;
                }
                ridx = range.index[k];
                spec = *sIt;
            }
        }
    }

    for (; k < range.number; ++k) {
        out += ptr[range.index[k]] * range.value[k];
    }
    for (; sIt != sEnd; ++sIt) {
        out += ptr[*sIt] * static_cast<Value_>(0);
    }

    return out;
}

template<typename Value_, typename Index_, typename Output_>
std::vector<tatami_stats::LocalOutputBuffer<Value_> > create_stores(size_t NR, size_t rhs_col, size_t thread, Index_ start, Index_ length, Output_* output) {
    std::vector<tatami_stats::LocalOutputBuffer<Value_> > stores;
    stores.reserve(rhs_col);
    size_t out_offset = 0; // using offsets instead of directly adding the pointer, to avoid forming an invalid address on the final iteration.
    for (size_t j = 0; j < rhs_col; ++j, out_offset += NR) {
        stores.emplace_back(thread, start, length, output + out_offset);
    }
    return stores;
}

}

}

#endif
