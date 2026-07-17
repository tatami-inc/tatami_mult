#ifndef TATAMI_MULT_DENSE_MATRIX_DENSE_ROW_ROW_TO_ROW_HPP
#define TATAMI_MULT_DENSE_MATRIX_DENSE_ROW_ROW_TO_ROW_HPP

#include <vector>
#include <cstddef>
#include <optional>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../../utils.hpp"

/**
 * @file row_to_row.hpp
 * @brief Dense row-major LHS, dense row-major matrix RHS, row-major output.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/dense_row/dense_matrix
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_dense_row_with_dense_row_matrix_to_row_output()`.
 */
struct MultiplyDenseRowWithDenseRowMatrixToRowOutputOptions {
    /**
     * Number of threads to use.
     * Different numbers of threads will not change the results. 
     */
    int num_threads = 1;

    /**
     * Primary block size, i.e., the number of LHS rows to be loaded at once.
     * This is also used to define the number of LHS columns in each block.
     * See the \f$B\f$ parameter in the @ref dense-blocking "Blocking for dense matrices" section for more details.
     */
    int primary_block_size = 16;

    /**
     * Secondary block size, i.e., the number of RHS columns to be processed in each block.
     * See the \f$C\f$ parameter in the @ref dense-blocking "Blocking for dense matrices" section for more details.
     * Different secondary block sizes will not change the results.
     */
    int secondary_block_size = 64;
};

/**
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightValue_ Numeric type of the RHS matrix value.
 * @tparam RightIndex_ Integer type of the RHS matrix index.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * @param right RHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * The number of rows in this matrix should be equal to the number of columns in `left`.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right` in row-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_dense_row_with_dense_row_matrix_to_row_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplyDenseRowWithDenseRowMatrixToRowOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right.ncol();

    auto right_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(common_dim);
    auto right_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(common_dim);
    populate_dense_buffers(true, common_dim, right_NC, right, right_buffers, right_ptrs, options.num_threads);

    const bool do_parallel = options.num_threads > 1;
    if (!do_parallel) {
        std::fill_n(output, sanisizer::product_unsafe<std::size_t>(right_NC, left_NR), 0);
    }

    if (options.primary_block_size == 1) {
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto ext = tatami::consecutive_extractor<false>(left, true, start, length);
            auto buffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(common_dim);

            // Use a temporary buffer for each output row to mitigate false sharing during updates across all 'c'.
            // There is still some false sharing when we transfer the results to the output row,
            // but this is fine as it is outside of the innermost loop.
            std::optional<std::vector<Output_> > tmp_output;
            if (do_parallel) {
                tmp_output.emplace(tatami::cast_Index_to_container_size<std::vector<Output_> >(right_NC));
            }

            for (LeftIndex_ lr = 0; lr < length; ++lr) {
                const auto left_ptr = ext->fetch(buffer.data());
                const auto optr = output + sanisizer::product_unsafe<std::size_t>(start + lr, right_NC);

                Output_* tmp_optr;
                if (!do_parallel) {
                    tmp_optr = optr;
                } else {
                    tmp_optr = tmp_output->data();
                }

                for (LeftIndex_ cd = 0; cd < common_dim; ++cd) {
                    const Output_ mult = left_ptr[cd];
                    const auto rightrow = right_ptrs[cd];
                    for (RightIndex_ rc = 0; rc < right_NC; ++rc) {
                        tmp_optr[rc] += static_cast<Output_>(rightrow[rc]) * mult;
                    }
                }

                if (do_parallel) {
                    std::copy_n(tmp_optr, right_NC, optr);
                    std::fill_n(tmp_optr, right_NC, 0);
                }
            }
        }, left_NR, options.num_threads);

    } else {
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto left_ext = tatami::consecutive_extractor<false>(left, true, start, length);
            std::vector<std::vector<LeftValue_> > left_buffers;
            std::vector<const LeftValue_*> left_ptrs;

            std::optional<std::vector<Output_> > tmp_output;
            {
                const LeftIndex_ max_block_rows = sanisizer::min(length, options.primary_block_size);
                left_buffers.reserve(max_block_rows);
                for (LeftIndex_  b = 0; b < max_block_rows ; ++b) {
                    left_buffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(common_dim));
                }
                sanisizer::resize(left_ptrs, max_block_rows);

                // Creating a block to hold the output during the updates over all 'c', to avoid false sharing.
                // We should be able to hold the block size in a size_t safely here, as this block is no larger than the array referenced by 'output'.
                // Of course, we might get an error if the vector's size_type is smaller than size_t but that seems a bit pathological.
                if (do_parallel) {
                    tmp_output.emplace(sanisizer::product<I<decltype(tmp_output->size())> >(max_block_rows, right_NC));
                }
            }

            LeftIndex_ lr = 0;
            while (lr < length) {
                const LeftIndex_ lr_num = sanisizer::min(options.primary_block_size, length - lr);
                for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                    left_ptrs[lr_counter] = left_ext->fetch(left_buffers[lr_counter].data());
                }

                Output_* const optr = output + sanisizer::product_unsafe<std::size_t>(start + lr, right_NC);
                Output_* tmp_optr;
                if (!do_parallel) {
                    tmp_optr = optr;
                } else {
                    tmp_optr = tmp_output->data();
                }

                LeftIndex_ cd = 0;
                while (cd < common_dim) { 
                    const LeftIndex_ cd_end = cd + sanisizer::min(options.primary_block_size, common_dim - cd);
                    RightIndex_ rc = 0;
                    while (rc < right_NC) {
                        const RightIndex_ rc_end = rc + sanisizer::min(options.secondary_block_size, right_NC - rc);

                        for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                            const auto matrow = left_ptrs[lr_counter];
                            const auto prod = tmp_optr + sanisizer::product_unsafe<std::size_t>(lr_counter, right_NC);
                            for (auto ccopy = cd; ccopy < cd_end; ++ccopy) {
                                const auto mult = matrow[ccopy];
                                const auto& rightrow = right_ptrs[ccopy];
                                for (auto rc_copy = rc; rc_copy < rc_end; ++rc_copy) {
                                    prod[rc_copy] += mult * rightrow[rc_copy];
                                }
                            }
                        }

                        rc = rc_end;
                    }
                    cd = cd_end;
                }

                if (do_parallel) {
                    const auto out_space = sanisizer::product_unsafe<std::size_t>(lr_num, right_NC);
                    std::copy_n(tmp_optr, out_space, optr);
                    std::fill_n(tmp_optr, out_space, 0);
                }
                lr += lr_num;
            }
        }, left_NR, options.num_threads);
    }
}

}

#endif
