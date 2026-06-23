#ifndef TATAMI_MULT_SPARSE_ROW_HPP
#define TATAMI_MULT_SPARSE_ROW_HPP

#include "dense_row.hpp"
#include "utils.hpp"

#include <vector>
#include <cstddef>
#include <type_traits>
#include <algorithm>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

namespace tatami_mult {

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

template<typename Value_, typename Index_, typename Right_, typename GetOutput_>
void sparse_row_dense_vectors(
    const tatami::Matrix<Value_, Index_>& matrix,
    const std::size_t num_rhs,
    const bool rhs_columnar,
    const std::vector<Right_*>& rhs_ptrs,
    const bool output_columnar,
    GetOutput_ get_output,
    const int num_threads
) {
    const Index_ NR = matrix.nrow();
    const Index_ NC = matrix.ncol();
    typedef I<decltype(get_output(0)[0])> Output;

    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(&matrix, true, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);

        for (Index_ r = 0; r < length; ++r) {
            const auto range = ext->fetch(vbuffer.data(), ibuffer.data());

            if (rhs_columnar) {
                if (output_columnar) {
                    for (I<decltype(num_rhs)> j = 0; j < num_rhs; ++j) {
                        const auto rptr = rhs_ptrs[j];
                        auto&& optr = get_output(j);
                        // Some false sharing potential here, but we just touch each location once per outer loop, so it's fine.
                        optr[start + r] = dense_sparse_dot_product<Output>(rptr, range);
                    }
                } else {
                    auto&& optr = get_output(start + r);
                    for (I<decltype(num_rhs)> j = 0; j < num_rhs; ++j) {
                        const auto rptr = rhs_ptrs[j];
                        optr[j] = dense_sparse_dot_product<Output>(rptr, range);
                    }
                }
            } else {
                if (output_columnar) {
                    for (Index_ x = 0; x < range.number; ++x) { 
                        const auto rptr = rhs_ptrs[start + range.index[x]];
                        const auto mult = range.value[x];
                        for (I<decltype(num_rhs)> j = 0; j < num_rhs; ++j) {
                            // TODO: not very cache-friendly, plus greater risk of false sharing as we're touching it inside the hottest loop.
                            // We might be able to fix this with some blocking. 
                            auto&& optr = get_output(j);
                            optr[start + r] += mult * rptr[j];
                        }
                    }
                } else {
                    auto&& optr = get_output(start + r);
                    for (Index_ x = 0; x < range.number; ++x) { 
                        const auto rptr = rhs_ptrs[start + range.index[x]];
                        const auto mult = range.value[x];
                        for (I<decltype(num_rhs)> j = 0; j < num_rhs; ++j) {
                            optr[j] += mult * rptr[j];
                        }
                    }
                }
            }
        }
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename Right_, typename Output_>
void sparse_row_vectors(const tatami::Matrix<Value_, Index_>& matrix, const std::vector<Right_*>& rhs_ptrs, const std::vector<Output_*>& output, int num_threads) {
    sparse_row_dense_vectors(
        matrix,
        rhs_ptrs.size(),
        true,
        rhs_ptrs,
        true,
        [&](const std::size_t c) -> Output_* { return output[c]; },
        num_threads
    );
}

template<typename Value_, typename Index_, typename RightValue_, typename RightIndex_, typename Output_>
void sparse_row_tatami_dense(
    const tatami::Matrix<Value_, Index_>& matrix,
    const tatami::Matrix<RightValue_, RightIndex_>& rhs,
    const bool output_columnar,
    Output_* const output,
    int num_threads
) {
    const auto rhs_row = rhs.nrow();
    const auto num_rhs = rhs.ncol();

    // The general strategy here is to drag the (hopefully small-ish) RHS matrix into memory,
    // avoiding the need to repeatedly use tatami to iterate over it. 

    if (!rhs.prefer_rows() || output_columnar) {
        // For row-major RHS matrices, we forcibly extract columns if the output is also columnar.
        // This avoids a suboptimal extraction within sparse_row_dense_vectors(), see the comments above.
        // we consider a one-time suboptimal extraction to be cheaper than many suboptimal accesses for multiplication itself.
        auto all_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(num_rhs);
        auto all_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(num_rhs);
        populate_dense_buffers(false, num_rhs, rhs_row, rhs, all_buffers, all_ptrs, num_threads);

        if (output_columnar) {
            const auto NR = matrix.nrow();
            sparse_row_dense_vectors(
                matrix,
                num_rhs,
                true,
                all_ptrs,
                true,
                [&](const std::size_t c) -> Output_* { return output + sanisizer::product_unsafe<std::size_t>(c, NR); },
                num_threads
            );
        } else {
            sparse_row_dense_vectors(
                matrix,
                num_rhs,
                true,
                all_ptrs,
                false,
                [&](const std::size_t r) -> Output_* { return output + sanisizer::product_unsafe<std::size_t>(r, num_rhs); },
                num_threads
            );
        }

    } else {
        auto all_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(rhs_row);
        auto all_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(rhs_row);
        populate_dense_buffers(true, rhs_row, num_rhs, rhs, all_buffers, all_ptrs, num_threads);

        sparse_row_dense_vectors(
            matrix,
            num_rhs,
            false,
            all_ptrs,
            false,
            [&](const std::size_t r) -> Output_* { return output + sanisizer::product_unsafe<std::size_t>(r, num_rhs); },
            num_threads
        );
    }
}

template<typename Value_, typename Index_, typename RightValue_, typename RightIndex_, typename Output_>
void sparse_row_tatami_sparse(
    const tatami::Matrix<Value_, Index_>& matrix,
    const tatami::Matrix<RightValue_, RightIndex_>& rhs,
    const bool output_columnar,
    Output_* const output,
    int num_threads
) {
    // We expand each sparse row into a dense format and then iterate over the sparse RHS columns.
    // So, the code is basically the same as that of the dense row-major matrices.
    dense_row_tatami_sparse(matrix, rhs, output_columnar, output, num_threads);
}

}

#endif
