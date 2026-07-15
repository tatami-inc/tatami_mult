#ifndef TATAMI_MULT_UTILS_HPP
#define TATAMI_MULT_UTILS_HPP

#include <type_traits>
#include <vector>
#include <cassert>

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

template<typename Index_>
struct FetchNonEmptySparseBlockInfo {
    FetchNonEmptySparseBlockInfo(const Index_ position, const Index_ num_non_empty, const bool consecutive) : 
        position(position), num_non_empty(num_non_empty), consecutive(consecutive) {}
    Index_ position, num_non_empty;
    bool consecutive;
};

template<typename Value_, typename Index_, class Zero_>
FetchNonEmptySparseBlockInfo<Index_> fetch_non_empty_sparse_block(
    tatami::OracularSparseExtractor<Value_, Index_>& ext,
    std::vector<std::vector<Value_> >& vbuffers,
    std::vector<std::vector<Index_> >& ibuffers,
    std::vector<tatami::SparseRange<Value_, Index_> >& ranges,
    std::vector<Index_>& non_empty,
    Index_ position,
    const Index_ length,
    const int block_size,
    Zero_ zero
) { 
    non_empty.clear();
    Index_ num_non_empty = 0;
    bool consecutive = true;

    // We assume that a check for remaining dimension elements was performed before continuing the blocked loop.
    assert(position < length);

    do {
        auto lrange = ext.fetch(vbuffers[num_non_empty].data(), ibuffers[num_non_empty].data());
        if (lrange.number == 0) {
            zero(position);
            consecutive = false;
            ++position;
            continue;
        }

        ranges[num_non_empty] = std::move(lrange);
        non_empty.push_back(position);
        ++num_non_empty;
        ++position;

        if (sanisizer::is_equal(num_non_empty, block_size)) {
            break;
        }
    } while (position < length);

    return FetchNonEmptySparseBlockInfo(position, num_non_empty, consecutive);
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
