#ifndef TATAMI_MULT_DENSE_MATRIX_DENSE_ROW_COLUMN_TO_ROW_HPP
#define TATAMI_MULT_DENSE_MATRIX_DENSE_ROW_COLUMN_TO_ROW_HPP

#include <vector>
#include <cstddef>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../../dense_dot_product.hpp"
#include "../../utils.hpp"

/**
 * @file column_to_row.hpp
 * @brief Dense row LHS, dense column-major matrix RHS, row-major output.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_dense_row_with_dense_column_matrix_to_row_output()`.
 */
struct MultiplyDenseRowWithDenseColumnMatrixToRowOutputOptions {
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
 * @tparam LeftValue_ Numeric type of the left matrix value.
 * @tparam LeftIndex_ Integer type of the left matrix index.
 * @tparam RightValue_ Numeric type of the right matrix value.
 * @tparam RightIndex_ Integer type of the right matrix index.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * @param right RHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer column access, but will work with all matrices.
 * The number of rows in this matrix should be equal to the number of columns in `left`.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right` in row-major format.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_dense_row_with_dense_column_matrix_to_row_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplyDenseRowWithDenseColumnMatrixToRowOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right.ncol();

    auto right_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(right_NC);
    auto right_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(right_NC);
    populate_dense_buffers(false, right_NC, common_dim, right, right_buffers, right_ptrs, options.num_threads);

    if (options.primary_block_size == 1) {
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto lext = tatami::consecutive_extractor<false>(left, true, start, length);
            auto lbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(common_dim);
            for (LeftIndex_ lr = 0; lr < length; ++lr) {
                const auto lptr = lext->fetch(lbuffer.data());
                for (RightIndex_ rc = 0; rc < right_NC; ++rc) {
                    // cast of common_dim to size_t is safe due to tatami's contract.
                    const auto res = dense_dot_product<accumulators_>(common_dim, lptr, right_ptrs[rc], static_cast<Output_>(0));
                    output[sanisizer::nd_offset<std::size_t>(rc, right_NC, start + lr)] = res;
                }
            }
        }, left_NR, options.num_threads);
        return;
    } 

    const bool do_parallel = options.num_threads > 1;
    if (!do_parallel) {
        std::fill_n(output, sanisizer::product_unsafe<std::size_t>(left_NR, right_NC), 0);
    }

    tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
        auto ext = tatami::consecutive_extractor<false>(left, true, start, length);

        const LeftIndex_ max_block_rows = sanisizer::min(length, options.primary_block_size);
        std::vector<std::vector<LeftValue_> > lbuffers;
        lbuffers.reserve(max_block_rows);
        for (LeftIndex_ b = 0; b < max_block_rows; ++b) {
            lbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(common_dim));
        }
        auto lptrs = tatami::create_container_of_Index_size<std::vector<const LeftValue_*> >(max_block_rows);

        std::optional<std::vector<Output_> > tmp_output;
        Output_* optr;
        if (do_parallel) {
            // For the multi-threaded case, we create some temporary buffers to hold the partial dot products for the current set of submatrices.
            // This aims to mitigate false sharing as we update each block's partial dot products in the loop over the common dimension.
            // There is still some potential for false sharing when we transfer the results to the output buffers,
            // but this is the same as the unblocked case so we won't worry about it.
            const auto max_block_cols = sanisizer::min(right_NC, options.primary_block_size);
            const auto max_block_rows = sanisizer::min(length, options.primary_block_size);
            tmp_output.emplace(sanisizer::product<I<decltype(tmp_output->size())> >(max_block_cols, max_block_rows));
            optr = tmp_output->data();
        } else {
            optr = output; // no need to add 'start' as this is zero in the single-threaded case.
        }

        LeftIndex_ lr = 0;
        while (lr < length) {
            const LeftIndex_ lr_num = sanisizer::min(options.primary_block_size, length - lr);
            for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                lptrs[lr_counter] = ext->fetch(lbuffers[lr_counter].data());
            }

            RightIndex_ rc = 0;
            while (rc < right_NC) {
                const RightIndex_ rc_num = sanisizer::min(options.primary_block_size, right_NC - rc);

                LeftIndex_ out_row_offset;
                RightIndex_ out_col_offset, out_stride;
                if (do_parallel) {
                    std::fill_n(optr, sanisizer::product_unsafe<std::size_t>(rc_num, lr_num), 0);
                    out_row_offset = 0;
                    out_col_offset = 0;
                    out_stride = rc_num;
                } else {
                    out_row_offset = lr; // start == 0 in serial code, so no need to add start here.
                    out_col_offset = rc;
                    out_stride = right_NC;
                }

                LeftIndex_ cd = 0;
                while (cd < common_dim) {
                    const LeftIndex_ cd_num = sanisizer::min(options.secondary_block_size, common_dim - cd);
                    for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                        const auto leftrow = lptrs[lr_counter];
                        for (RightIndex_ rc_counter = 0; rc_counter < rc_num; ++rc_counter) {
                            auto& dest = optr[sanisizer::nd_offset<std::size_t>(out_col_offset + rc_counter, out_stride, out_row_offset + lr_counter)]; 
                            dest = dense_dot_product<accumulators_>(
                                cd_num, // cast to size_t is safe due to tatami's contract.
                                right_ptrs[rc + rc_counter] + cd,
                                leftrow + cd,
                                dest
                            );
                        }
                    }
                    cd += cd_num;
                }

                if (do_parallel) {
                    for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                        std::copy_n(
                            optr + sanisizer::product_unsafe<std::size_t>(rc_num, lr_counter),
                            rc_num,
                            output + sanisizer::nd_offset<std::size_t>(rc, right_NC, start + lr + lr_counter)
                        );
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
