#ifndef TATAMI_MULT_DENSE_DOT_PRODUCT_H
#define TATAMI_MULT_DENSE_DOT_PRODUCT_H

#include <cstddef>
#include <array>
#include <numeric>

namespace tatami_mult {

// Force unrolling to avoid relying on optimizer decisions.
// For example, older versions of GCC won't unroll at -O2, so we need to do it ourselves if we don't want a run-time nested loop.
// Fortunately, compilers will stil auto-vectorize a manually-unrolled loop, so this won't be a pessimisation in the long term.
template<std::size_t counter_ = 0, class Iterator1_, class Iterator2_, typename Output_, std::size_t accumulators_>
void unrolled_dense_dot_product(const std::size_t idx, Iterator1_ start1, Iterator2_ start2, std::array<Output_, accumulators_>& dots) {
    dots[counter_] += static_cast<Output_>(*(start1 + idx + counter_)) * static_cast<Output_>(*(start2 + idx + counter_));
    if constexpr(counter_ + 1 < accumulators_) {
        unrolled_dense_dot_product<counter_ + 1>(idx, start1, start2, dots);
    }
}

template<std::size_t accumulators_, class Iterator1_, class Iterator2_, typename Output_>
Output_ dense_dot_product(const std::size_t len, Iterator1_ start1, Iterator2_ start2, Output_ initial) {
    if constexpr(accumulators_ == 1) {
        Output_ dot = initial;
        for (std::size_t i = 0; i < len; ++i) {
            dot += static_cast<Output_>(*(start1 + i)) * static_cast<Output_>(*(start2 + i));
        }
        return dot;

    } else {
        const std::size_t cycles = len / accumulators_;
        const std::size_t remainder = len % accumulators_;
        std::array<Output_, accumulators_> dots{};

        // One might think to create a peeling iteration to initialize the array with the first cycle of the dot products, like:
        // 
        // for (std::size_t i = 0; i < accumulators_; ++i) {
        //     dots[i] = *(start1 + i) * *(start2 + i);
        // }
        // counter = accumulators_;
        // 
        // and setting c = 1 below, which saves us one set of additions.
        // But when I test this out, this seems to be no better, and indeed, a slight pessimisation for len ~= 200.
        // I would guess that the time saved by skipping the addition is offset by the increased size of the program.

        for (std::size_t c = 0; c < cycles; ++c) {
            unrolled_dense_dot_product(
                c * accumulators_, // this is less than 'len' and so must fit in a std::size_t.
                start1,
                start2,
                dots
            );
        }

        // Keep adding to 'dots' to provide more opportunities for auto-vectorization.
        for (std::size_t i = 0; i < remainder; ++i) {
            const auto idx = cycles * accumulators_ + i;
            dots[i] += *(start1 + idx) * *(start2 + idx);
        }

        return std::accumulate(dots.begin(), dots.end(), initial);
    }
}

}

#endif
