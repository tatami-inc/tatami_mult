#ifndef TATAMI_MULT_SPARSE_ROW_HPP
#define TATAMI_MULT_SPARSE_ROW_HPP

#include "utils.hpp"

#include <vector>
#include <cstddef>
#include <type_traits>
#include <algorithm>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

namespace tatami_mult {

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

template<typename Value_, typename Index_, typename Right_, typename Output_>
void sparse_row_vector(const tatami::Matrix<Value_, Index_>& matrix, const Right_* rhs, Output_* output, int num_threads) {
    const Index_ NR = matrix.nrow();
    const Index_ NC = matrix.ncol();
    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(&matrix, true, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);
        for (Index_ r = start, end = start + length; r < end; ++r) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());
            output[r] = dense_sparse_dot_product<Output_>(rhs, range);
        }
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename Right_, typename Output_>
void sparse_row_vectors(const tatami::Matrix<Value_, Index_>& matrix, const std::vector<Right_*>& rhs, const std::vector<Output_*>& output, int num_threads) {
    const Index_ NR = matrix.nrow();
    const Index_ NC = matrix.ncol();
    const auto num_rhs = rhs.size();
    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(&matrix, true, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);
        for (Index_ r = start, end = start + length; r < end; ++r) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());
            for (I<decltype(num_rhs)> j = 0; j < num_rhs; ++j) {
                output[j][r] = dense_sparse_dot_product<Output_>(rhs[j], range);
            }
        }
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename RightValue_, typename RightIndex_, typename Output_>
void sparse_row_tatami_dense(
    const tatami::Matrix<Value_, Index_>& matrix,
    const tatami::Matrix<RightValue_, RightIndex_>& rhs,
    Output_* output,
    RightIndex_ row_shift,
    Index_ col_shift,
    int num_threads
) {
    const Index_ NR = matrix.nrow();
    const Index_ NC = matrix.ncol();
    const RightIndex_ rhs_col = rhs.ncol();

    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(&matrix, true, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);
        auto rbuffer = tatami::create_container_of_Index_size<std::vector<RightValue_> >(NC);

        for (Index_ r = start, end = start + length; r < end; ++r) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());
            auto rext = tatami::consecutive_extractor<false>(&rhs, false, static_cast<RightIndex_>(0), rhs_col);
            auto start_offset = sanisizer::product_unsafe<std::size_t>(r, row_shift); // offsets must fit in size_t if output is correctly sized.
            for (RightIndex_ j = 0; j < rhs_col; ++j) {
                auto rptr = rext->fetch(rbuffer.data());
                auto cur_offset = start_offset + sanisizer::product_unsafe<std::size_t>(j, col_shift);
                output[cur_offset] = dense_sparse_dot_product<Output_>(rptr, range);
            }
        }
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename RightValue_, typename RightIndex_, typename Output_>
void sparse_row_tatami_sparse(
    const tatami::Matrix<Value_, Index_>& matrix,
    const tatami::Matrix<RightValue_, RightIndex_>& rhs,
    Output_* output,
    RightIndex_ row_shift,
    Index_ col_shift,
    int num_threads)
{
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();
    RightIndex_ rhs_col = rhs.ncol();

    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<false>(&matrix, true, start, length);
        auto expanded = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
        auto rvbuffer = tatami::create_container_of_Index_size<std::vector<RightValue_> >(NC);
        auto ribuffer = tatami::create_container_of_Index_size<std::vector<RightIndex_> >(NC);

        for (Index_ r = start, end = start + length; r < end; ++r) {
            auto ptr = ext->fetch(expanded.data());
            auto rext = tatami::consecutive_extractor<true>(&rhs, false, static_cast<RightIndex_>(0), rhs_col);
            auto start_offset = sanisizer::product_unsafe<std::size_t>(r, row_shift); // offsets must fit in size_t if output is correctly sized.
            for (RightIndex_ j = 0; j < rhs_col; ++j) {
                auto rrange = rext->fetch(rvbuffer.data(), ribuffer.data());
                auto cur_offset = start_offset + sanisizer::product_unsafe<std::size_t>(j, col_shift);
                output[cur_offset] = dense_sparse_dot_product<Output_>(ptr, rrange);
            }
        }
    }, NR, num_threads);
}

}

#endif
