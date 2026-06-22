#ifndef TATAMI_MULT_UTILS_HPP
#define TATAMI_MULT_UTILS_HPP

#include <type_traits>
#include <vector>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

namespace tatami_mult {

template<typename Input_>
using I = std::remove_cv_t<std::remove_reference_t<Input_> >;

template<typename Value_, typename Index_>
void populate_dense_buffers(
    const bool row,
    const Index_ primary,
    const Index_ secondary,
    const tatami::Matrix<Value_, Index_>& matrix,
    std::vector<std::vector<Value_> >& all_buffers, // pass by reference so that a move won't invalidate the pointers.
    std::vector<const Value_*>& all_ptrs,
    int num_threads
) {
    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        std::vector<Value_> tmp;
        sanisizer::reserve(tmp, secondary);
        auto ext = tatami::consecutive_extractor<false>(matrix, row, start, length);
        for (Index_ i = start, end = start + length; i < end; ++i) {
            tmp.resize(secondary); // reserve is safe, so should resize.
            auto ptr = ext->fetch(tmp.data());
            if (ptr == tmp.data()) {
                all_buffers[i].swap(tmp);
                all_ptrs[i] = all_buffers[i].data(); // only do this after the swap to ensure pointer validity.
            } else {
                all_ptrs[i] = ptr;
            }
        }
    }, primary, num_threads);
}

template<typename Value_, typename Index_>
void populate_sparse_buffers(
    const bool row,
    const Index_ primary,
    const Index_ secondary,
    const tatami::Matrix<Value_, Index_>& matrix,
    std::vector<std::vector<Value_> >& all_vbuffers,
    std::vector<std::vector<Index_> >& all_ibuffers,
    std::vector<tatami::SparseRange<Value_, Index_> >& all_ranges,
    int num_threads
) {
    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        std::vector<Value_> tmp_v;
        sanisizer::reserve(tmp_v, secondary);
        std::vector<Index_> tmp_i;
        sanisizer::reserve(tmp_i, secondary);

        auto ext = tatami::consecutive_extractor<true>(matrix, row, start, length);
        for (Index_ i = start, end = start + length; i < end; ++i) {
            tmp_v.resize(secondary); // reserve is safe, so should resize.
            tmp_i.resize(secondary);

            auto range = ext->fetch(tmp_v.data(), tmp_i.data());
            if (range.value == tmp_v.data()) {
                all_vbuffers[i].swap(tmp_v);
                range.value = all_vbuffers[i].data();
            }
            if (range.index == tmp_i.data()) {
                all_ibuffers[i].swap(tmp_i);
                range.index = all_ibuffers[i].data();
            }
            all_ranges[i] = range;
        }
    }, primary, num_threads);
}

template<typename Output_, typename DenseValue_, typename Value_, typename Index_>
Output_ dense_sparse_dot_product(const DenseValue_* ptr, const tatami::SparseRange<Value_, Index_>& range) {
    if (range.number == 0) {
        return 0;
    }

    // Copying Eigen's use of two accumulators; effectively unrolls the loop a little for speed.
    Output_ dot1 = 0, dot2 = 0;

    Index_ s = 0;
    const Index_ number_m1 = range.number - 1;
    for (; s < number_m1; s += 2) {
        dot1 += range.value[s] * ptr[range.index[s]];
        dot2 += range.value[s + 1] * ptr[range.index[s + 1]];
    }

    if (s < range.number) {
        dot1 += range.value[s] * ptr[range.index[s]];
    }
    return dot1 + dot2;
}

}

#endif
