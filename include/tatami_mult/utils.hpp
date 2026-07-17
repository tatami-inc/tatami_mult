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

template<typename Index_>
struct FetchNonEmptySparseBlockInfo {
    FetchNonEmptySparseBlockInfo(const Index_ position, const Index_ num_non_empty, const bool all_non_empty) : 
        position(position), num_non_empty(num_non_empty), all_non_empty(all_non_empty) {}
    Index_ position, num_non_empty;
    bool all_non_empty;
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
    bool all_non_empty = true;

    // We assume that a check for remaining dimension elements was performed before continuing the blocked loop.
    assert(position < length);

    do {
        auto lrange = ext.fetch(vbuffers[num_non_empty].data(), ibuffers[num_non_empty].data());
        if (lrange.number == 0) {
            zero(position);
            all_non_empty = false;
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

    return FetchNonEmptySparseBlockInfo(position, num_non_empty, all_non_empty);
}

}

#endif
