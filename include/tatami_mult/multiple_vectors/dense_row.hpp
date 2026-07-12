#ifndef TATAMI_MULT_MULTIPLE_VECTORS_DENSE_ROW_HPP
#define TATAMI_MULT_MULTIPLE_VECTORS_DENSE_ROW_HPP

#include <cstddef>
#include <vector>
#include <optional>

#include "tatami/tatami.hpp"

#include "../utils.hpp"
#include "../dense_dot_product.hpp"

/**
 * @file dense_row.hpp
 * @brief Dense row-major LHS, multiple vectors RHS.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_dense_row_with_multiple_vectors()`.
 */
struct MultiplyDenseRowWithMultipleVectorsOptions {
    /**
     * Number of threads to use.
     */
    int num_threads = 1;

    /**
     * Primary block size.
     */
    int primary_block_size = 16;

    /**
     * Secondary block size.
     */
    int secondary_block_size = 64;
};

/**
 * @tparam accumulators_ Number of accumulators for computing the dot product.
 * This should be positive and is very often a power of 2, with values of 2-8 typically providing some performance improvement on modern CPUs.
 * Different numbers of accumulators may result in slight changes to the output due to changes in floating-point round-off error.
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the matrix index.
 * @tparam Right_ Numeric type of the vector on the right hand side.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * @param[in] right Vector of pointers, each of which points to an array of length `left.ncol()`.
 * Each entry contains a vector with which to multiply `left`.
 * @param[out] output Vector of pointers, each of which points to an array of length `left.nrow()`.
 * On output, the `i`-th entry stores the product `left * right[i]`.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename Value_, typename Index_, typename Right_, typename Output_>
void multiply_dense_row_with_multiple_vectors(
    const tatami::Matrix<Value_, Index_>& left,
    const std::vector<Right_*>& right,
    const std::vector<Output_*>& output,
    const MultiplyDenseRowWithMultipleVectorsOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right.size();
    typedef I<decltype(right_NC)> RightIndex;

    if (options.primary_block_size == 1) {
        tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
            auto lext = tatami::consecutive_extractor<false>(left, true, start, length);
            auto lbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(common_dim);
            for (Index_ lr = 0; lr < length; ++lr) {
                const auto lptr = lext->fetch(lbuffer.data());
                for (RightIndex rc = 0; rc < right_NC; ++rc) {
                    // Implicit cast of common_dim to std::size_t is safe, as per the tatami contract.
                    output[rc][start + lr] = dense_dot_product<accumulators_>(common_dim, lptr, right[rc], static_cast<Output_>(0));
                }
            }
        }, left_NR, options.num_threads);
        return;
    } 

    const bool do_parallel = options.num_threads > 1;
    if (!do_parallel) {
        // Zeroing all of the buffers if we're operating on a single thread.
        for (RightIndex h = 0; h < right_NC; ++h) {
            std::fill_n(output[h], left_NR, 0);
        }
    }

    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<false>(left, true, start, length);

        const Index_ max_block_rows = sanisizer::min(length, options.primary_block_size);
        std::vector<std::vector<Value_> > left_buffers;
        left_buffers.reserve(max_block_rows);
        for (Index_ lr = 0; lr < max_block_rows; ++lr) {
            left_buffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<Value_> >(common_dim));
        }
        auto left_ptrs = tatami::create_container_of_Index_size<std::vector<const Value_*> >(max_block_rows);

        std::optional<std::vector<std::vector<Output_> > > tmp_output;
        std::optional<std::vector<Output_*> > tmp_outptrs;
        if (do_parallel) {
            // For the multi-threaded case, we create some temporary buffers to hold the partial dot products for the current set of submatrices.
            // This aims to mitigate false sharing as we update each block's partial dot products in the loop over the common dimension.
            // There is still some potential for false sharing when we transfer the results to the output buffers,
            // but this is the same as the unblocked case so we won't worry about it.
            const RightIndex max_block_cols = sanisizer::min(right_NC, options.primary_block_size);
            tmp_output.emplace();
            tmp_output->reserve(max_block_cols);
            for (RightIndex rc = 0; rc < max_block_cols; ++rc) {
                tmp_output->emplace_back(tatami::cast_Index_to_container_size<std::vector<Output_> >(max_block_rows));
            }
            tmp_outptrs.emplace();
            tmp_outptrs->reserve(max_block_cols);
            for (RightIndex rc = 0; rc < max_block_cols; ++rc) {
                tmp_outptrs->emplace_back((*tmp_output)[rc].data());
            }
        }

        Index_ lr = 0;
        while (lr < length) {
            const Index_ lr_num = sanisizer::min(options.primary_block_size, length - lr);
            for (Index_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                left_ptrs[lr_counter] = ext->fetch(left_buffers[lr_counter].data());
            }

            RightIndex rc = 0;
            while (rc < right_NC) {
                const RightIndex rc_num = sanisizer::min(options.primary_block_size, right_NC - rc);

                Output_* const * outptrs;
                Index_ out_row_offset;
                if (do_parallel) {
                    outptrs = tmp_outptrs->data();
                    out_row_offset = 0; // no need to add start, as it's ignored when saving to tmp_output.
                } else {
                    outptrs = output.data() + rc;
                    out_row_offset = lr; // no need to add start, as it's zero when there's only one thread. 
                }

                Index_ cd = 0;
                while (cd < common_dim) {
                    const Index_ cd_num = sanisizer::min(options.secondary_block_size, common_dim - cd);
                    for (RightIndex rc_counter = 0; rc_counter < rc_num; ++rc_counter) {
                        const auto outcol = outptrs[rc_counter];
                        const auto& rightcol = right[rc + rc_counter];
                        for (Index_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                            auto& dest = outcol[out_row_offset + lr_counter]; 
                            // Implicit cast of cd_num to std::size_t is safe, as per the tatami contract.
                            dest = dense_dot_product<accumulators_>(
                                cd_num,
                                rightcol + cd,
                                left_ptrs[lr_counter] + cd,
                                dest
                            );
                        }
                    }
                    cd += cd_num;
                }

                if (do_parallel) {
                    for (RightIndex rc_counter = 0; rc_counter < rc_num; ++rc_counter) {
                        auto& src = (*tmp_output)[rc_counter];
                        std::copy_n(src.begin(), lr_num, output[rc + rc_counter] + start + lr);
                        std::fill_n(src.begin(), lr_num, 0);
                    }
                }

                rc += rc_num;
            }
            lr += lr_num;
        }
    }, left_NR, options.num_threads);
}

}

#endif
