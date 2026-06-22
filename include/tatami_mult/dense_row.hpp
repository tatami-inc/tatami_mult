#ifndef TATAMI_MULT_DENSE_ROW_HPP
#define TATAMI_MULT_DENSE_ROW_HPP

#include "utils.hpp"

#include <vector>
#include <algorithm>
#include <cstddef>
#include <type_traits>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

namespace tatami_mult {

template<typename Value_, typename Index_, typename Right_, typename Output_>
void dense_row_vector(const tatami::Matrix<Value_, Index_>& matrix, const Right_* rhs, Output_* output, int num_threads) {
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();

    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<false>(&matrix, true, start, length);
        auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);

        for (Index_ r = start, end = start + length; r < end; ++r) {
            auto ptr = ext->fetch(buffer.data());
            output[r] = std::inner_product(ptr, ptr + NC, rhs, static_cast<Output_>(0));
        }

    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename Right_, typename GetOutput_>
void dense_row_dense_vectors(
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
        auto ext = tatami::consecutive_extractor<false>(&matrix, true, start, length);
        auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);

        for (Index_ r = 0; r < length; ++r) {
            const auto ptr = ext->fetch(buffer.data());

            if (rhs_columnar) {
                // TODO: wouldn't hurt to use a blocking strategy here either, to reduce cache misses.
                if (output_columnar) {
                    for (I<decltype(num_rhs)> j = 0; j < num_rhs; ++j) {
                        const auto rptr = rhs_ptrs[j];
                        auto&& optr = get_output(j);
                        optr[start + r] = std::inner_product(ptr, ptr + NC, rptr, static_cast<Output>(0));
                    }

                } else {
                    auto&& optr = get_output(start + r);
                    for (I<decltype(num_rhs)> j = 0; j < num_rhs; ++j) {
                        const auto rptr = rhs_ptrs[j];
                        optr[j] = std::inner_product(ptr, ptr + NC, rptr, static_cast<Output>(0));
                    }
                }

            } else {
                // TODO: collect values and use a block strategy here, lots of non-contiguous access in the hottest loop.
                if (output_columnar) {
                    for (I<decltype(num_rhs)> j = 0; j < num_rhs; ++j) {
                        Output dot = 0;
                        for (Index_ c = 0; c < NC; ++c) {
                            dot += rhs_ptrs[c][j] * ptr[c];
                        }
                        auto&& optr = get_output(j);
                        optr[start + r] = dot;
                    }

                } else {
                    auto&& optr = get_output(start + r);
                    for (I<decltype(num_rhs)> j = 0; j < num_rhs; ++j) {
                        Output dot = 0;
                        for (Index_ c = 0; c < NC; ++c) {
                            dot += rhs_ptrs[c][j] * ptr[c];
                        }
                        optr[j] = dot;
                    }
                }
            }
        }
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename Right_, typename Output_>
void dense_row_vectors(const tatami::Matrix<Value_, Index_>& matrix, const std::vector<Right_*>& rhs_ptrs, const std::vector<Output_*>& output, int num_threads) {
    dense_row_dense_vectors(
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
void dense_row_tatami_dense(
    const tatami::Matrix<Value_, Index_>& matrix,
    const tatami::Matrix<RightValue_, RightIndex_>& rhs,
    Output_* output,
    RightIndex_ row_shift,
    Index_ col_shift,
    int num_threads
) {
    assert(row_shift == 1 || col_shift == 1);
    const auto rhs_row = rhs.nrow();
    const auto num_rhs = rhs.ncol();

    // The general strategy here is to drag the (hopefully small-ish) RHS matrix into memory,
    // avoiding the need to repeatedly use tatami to iterate over it. 

    if (rhs.prefer_rows()) {
        auto all_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(rhs_row);
        auto all_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(rhs_row);
        populate_dense_buffers(true, rhs_row, num_rhs, rhs, all_buffers, all_ptrs, num_threads);

        if (row_shift == 1) { // i.e., output is columnar.
            const auto NR = matrix.nrow();
            dense_row_dense_vectors(
                matrix,
                num_rhs,
                false,
                all_ptrs,
                true,
                [&](const std::size_t c) -> Output_* { return output + sanisizer::product_unsafe<std::size_t>(c, NR); },
                num_threads
            );

        } else {
            dense_row_dense_vectors(
                matrix,
                num_rhs,
                false,
                all_ptrs,
                false,
                [&](const std::size_t r) -> Output_* { return output + sanisizer::product_unsafe<std::size_t>(r, num_rhs); },
                num_threads
            );
        }

    } else {
        auto all_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(num_rhs);
        auto all_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(num_rhs);
        populate_dense_buffers(false, num_rhs, rhs_row, rhs, all_buffers, all_ptrs, num_threads);

        if (row_shift == 1) { // i.e., output is columnar.
            const auto NR = matrix.nrow();
            dense_row_dense_vectors(
                matrix,
                num_rhs,
                true,
                all_ptrs,
                true,
                [&](const std::size_t c) -> Output_* { return output + sanisizer::product_unsafe<std::size_t>(c, NR); },
                num_threads
            );
        } else {
            dense_row_dense_vectors(
                matrix,
                num_rhs,
                true,
                all_ptrs,
                false,
                [&](const std::size_t r) -> Output_* { return output + sanisizer::product_unsafe<std::size_t>(r, num_rhs); },
                num_threads
            );
        }
    }
}

template<typename Value_, typename Index_, typename RightValue_, typename RightIndex_, typename Output_>
void dense_row_tatami_sparse(
    const tatami::Matrix<Value_, Index_>& matrix,
    const tatami::Matrix<RightValue_, RightIndex_>& rhs,
    Output_* output,
    RightIndex_ row_shift,
    Index_ col_shift,
    int num_threads
) {
    const bool output_columnar = (row_shift == 1);
    assert(row_shift == 1 || col_shift == 1);
    const Index_ NR = matrix.nrow();
    const Index_ NC = matrix.ncol();
    assert(sanisizer::is_equal(NC, rhs.nrow()));
    const auto num_rhs = rhs.ncol();

    // The general strategy here is to drag the (hopefully small-ish) RHS matrix into memory,
    // avoiding the need to repeatedly use tatami to iterate over it. 
    // For row-major RHS matrices, we forcibly extract columns if the output is also columnar.
    // This avoids a suboptimal extraction within dense_row_sparse_vectors(), see the comments below.
    // We consider a one-time suboptimal extraction to be cheaper than many suboptimal accesses for multiplication itself.

    if (!rhs.prefer_rows() || output_columnar) {
        auto all_vbuffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(num_rhs);
        auto all_ibuffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightIndex_> > >(num_rhs);
        auto all_ranges = tatami::create_container_of_Index_size<std::vector<tatami::SparseRange<RightValue_, RightIndex_> > >(num_rhs);
        populate_sparse_buffers(false, num_rhs, NC, rhs, all_vbuffers, all_ibuffers, all_ranges, num_threads);

        if (output_columnar) {
            tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
                auto ext = tatami::consecutive_extractor<false>(matrix, true, start, length);
                auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
                for (Index_ r = start, end = start + length; r < end; ++r) {
                    auto ptr = ext->fetch(buffer.data());
                    for (I<decltype(num_rhs)> j = 0; j < num_rhs; ++j) {
                        const auto& range = all_ranges[j];
                        // Non-contiguous writes are a little annoying and more susceptible to false sharing,
                        // but they're also relatively infrequent, so we'll just accept it.
                        output[sanisizer::nd_offset<std::size_t>(r, NR, j)] = dense_sparse_dot_product<Output_>(ptr, range);
                    }
                }
            }, NR, num_threads);

        } else {
            tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
                auto ext = tatami::consecutive_extractor<false>(matrix, true, start, length);
                auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
                for (Index_ r = start, end = start + length; r < end; ++r) {
                    auto ptr = ext->fetch(buffer.data());
                    for (I<decltype(num_rhs)> j = 0; j < num_rhs; ++j) {
                        const auto& range = all_ranges[j];
                        output[sanisizer::nd_offset<std::size_t>(j, num_rhs, r)] = dense_sparse_dot_product<Output_>(ptr, range);
                    }
                }
            }, NR, num_threads);
        }

    } else {
        auto all_vbuffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(NC);
        auto all_ibuffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightIndex_> > >(NC);
        auto all_ranges = tatami::create_container_of_Index_size<std::vector<tatami::SparseRange<RightValue_, RightIndex_> > >(NC);
        populate_sparse_buffers(true, NC, num_rhs, rhs, all_vbuffers, all_ibuffers, all_ranges, num_threads);

        /* For columnar output, the per-thread loop would look something like this:

            for (Index_ r = start, end = start + length; r < end; ++r) {
                auto ptr = ext->fetch(buffer.data());
                for (Index_ c = 0; c < NC; ++c) {
                    auto range = get_rhs(c);
                    for (I<decltype(range.number)> x = 0; x < range.number; ++x) {
                        auto&& optr = get_output(range.index[x]);
                        optr[r] += ptr[range.index[x]] * range.value[x];
                    }
                }
            }

         * The innermost loop is pretty gross given the non-contiguous access. 
         * There is also a high risk of false sharing in a multi-threaded context.
         * We've got a sparse matrix so we can't easily use blocking; so, we'll avoid columnar output if we have a non-columnar RHS.
         */
        assert(!output_columnar);

        tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
            auto ext = tatami::consecutive_extractor<false>(matrix, true, start, length);
            auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
            for (Index_ r = start, end = start + length; r < end; ++r) {
                auto ptr = ext->fetch(buffer.data());
                for (Index_ c = 0; c < NC; ++c) {
                    const auto& range = all_ranges[c];
                    for (I<decltype(range.number)> x = 0; x < range.number; ++x) {
                        output[sanisizer::nd_offset<std::size_t>(range.index[x], num_rhs, r)] += ptr[range.index[x]] * range.value[x];
                    }
                }
            }
        }, NR, num_threads);
    }
}

}

#endif
