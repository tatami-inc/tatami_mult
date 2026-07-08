#ifndef TATAMI_MULT_SPARSE_DOT_PRODUCT_H
#define TATAMI_MULT_SPARSE_DOT_PRODUCT_H

#include <cstddef>
#include <array>
#include <numeric>

namespace tatami_mult {

// Force unrolling to avoid relying on optimizer decisions.
// For example, older versions of GCC won't unroll at -O2, so we need to do it ourselves if we don't want a run-time nested loop.
// Fortunately, compilers will stil auto-vectorize a manually-unrolled loop, so this won't be a pessimisation in the long term.
template<std::size_t accumulators_, std::size_t counter_ = 0, class ValueIterator_, class IndexIterator_, typename Dense_, typename Output_>
void unrolled_sparse_dot_product(const std::size_t idx, ValueIterator_ vptr, IndexIterator_ iptr, Dense_ dense, std::array<Output_, accumulators_>& dots) {
    dots[counter_] += static_cast<Output_>(dense[*(iptr + idx + counter_)]) * static_cast<Output_>(*(vptr + idx + counter_));
    if constexpr(counter_ + 1 < accumulators_) {
        unrolled_sparse_dot_product<accumulators_, counter_ + 1>(idx, vptr, iptr, dense, dots);
    }
}

template<std::size_t accumulators_, class ValueIterator_, class IndexIterator_, typename Dense_, typename Output_>
Output_ sparse_dot_product(const std::size_t num_non_zeros, ValueIterator_ vptr, IndexIterator_ iptr, Dense_ dense, Output_ initial) {
    if constexpr(accumulators_ == 1) {
        Output_ dot = initial;
        for (std::size_t i = 0; i < num_non_zeros; ++i) {
            dot += static_cast<Output_>(dense[*(iptr + i)]) * static_cast<Output_>(*(vptr + i));
        }
        return dot;

    } else {
        std::array<Output_, accumulators_> dots{};
        const std::size_t cycles = num_non_zeros / accumulators_;
        const std::size_t remainder = num_non_zeros % accumulators_;

        for (std::size_t c = 0; c < cycles; ++c) {
            unrolled_sparse_dot_product(c * accumulators_, vptr, iptr, dense, dots);
        }

        Output_ extras = initial;
        for (std::size_t i = 0; i < remainder; ++i) {
            const auto idx = cycles * accumulators_ + i;
            extras += dense[*(iptr + idx)] * *(vptr + idx);
        }
        return std::accumulate(dots.begin(), dots.end(), extras);
    }
}

}

#endif
