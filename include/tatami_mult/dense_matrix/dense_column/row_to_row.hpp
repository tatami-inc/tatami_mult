#ifndef TATAMI_MULT_DENSE_MATRIX_DENSE_COLUMN_ROW_TO_ROW_HPP
#define TATAMI_MULT_DENSE_MATRIX_DENSE_COLUMN_ROW_TO_ROW_HPP

#include <cstddef>
#include <vector>
#include <optional>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../../utils.hpp"

/**
 * @file row_to_row.hpp
 * @brief Dense column LHS, dense row-major matrix RHS, row-major output.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_dense_column_with_dense_row_matrix_to_row_output()`.
 */
struct MultiplyDenseColumnWithDenseRowMatrixToRowOutputOptions {
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
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the matrix index.
 * @tparam Right_ Numeric type of the vector on the right hand side.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer column access, but will work with all matrices.
 * @param right RHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this contains the product `left * right` in row-major order.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_dense_column_with_dense_row_matrix_to_row_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplyDenseColumnWithDenseRowMatrixToRowOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right.ncol();
    std::fill_n(output, sanisizer::product_unsafe<std::size_t>(left_NR, right_NC), 0);

    auto right_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(right_NC);
    auto right_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(right_NC);
    populate_dense_buffers(false, right_NC, common_dim, right, right_buffers, right_ptrs, options.num_threads);

    const bool do_parallel = options.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<Output_> > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(options.num_threads - 1));
    }

    const auto num_used = tatami::parallelize([&](int t, LeftIndex_ start, LeftIndex_ length) -> void {
        auto left_ext = tatami::consecutive_extractor<false>(left, false, start, length);
        auto right_ext = tatami::consecutive_extractor<false>(right, true, start, length);

        std::optional<std::vector<Output_> > tmp_output;
        Output_* outptr; 
        if (!do_parallel || t == 0) {
            outptr = output;
        } else {
            tmp_output.emplace(sanisizer::product<I<decltype(tmp_output->size())> >(left_NR, right_NC));
            outptr = tmp_output->data();
        }

        if (options.primary_block_size == 1) {
            auto left_buffer = tatami::create_container_of_Index_size<std::vector<Output_> >(left_NR);
            auto right_buffer = tatami::create_container_of_Index_size<std::vector<Output_> >(right_NC);
            for (LeftIndex_ cd = 0; cd < length; ++cd) {
                const auto left_ptr = left_ext->fetch(left_buffer.data());
                const auto right_ptr = right_ext->fetch(right_buffer.data());
                for (LeftIndex_ lr = 0; lr < left_NR; ++lr) {
                    const Output_ mult = left_ptr[lr];
                    for (RightIndex_ rc = 0; rc < right_NC; ++rc) {
                        outptr[sanisizer::nd_offset<std::size_t>(rc, right_NC, lr)] += mult * static_cast<Output_>(right_ptr[rc]);
                    }
                }
            }

        } else {
            std::vector<std::vector<LeftValue_> > left_buffers;
            std::vector<std::vector<RightValue_> > right_buffers;
            std::vector<const LeftValue_*> left_ptrs;
            std::vector<const RightValue_*> right_ptrs;
            {
                const LeftIndex_ max_block_cols = sanisizer::min(length, options.primary_block_size);
                left_buffers.reserve(max_block_cols);
                right_buffers.reserve(max_block_cols);
                for (LeftIndex_ cd = 0; cd < max_block_cols; ++cd) {
                    left_buffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(left_NR));
                    right_buffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<RightValue_> >(right_NC));
                }
                sanisizer::resize(left_ptrs, max_block_cols);
                sanisizer::resize(right_ptrs, max_block_cols);
            }

            LeftIndex_ cd = 0;
            while (cd < length) {
                const auto cd_num = sanisizer::min(options.primary_block_size, length - cd);
                for (LeftIndex_ cd_counter = 0; cd_counter < cd_num; ++cd_counter) {
                    left_ptrs[cd_counter] = left_ext->fetch(left_buffers[cd_counter].data());
                    right_ptrs[cd_counter] = right_ext->fetch(right_buffers[cd_counter].data());
                }

                LeftIndex_ lr = 0;
                while (lr < left_NR) {
                    const LeftIndex_ lr_end = lr + sanisizer::min(options.primary_block_size, left_NR - lr);
                    RightIndex_ rc = 0;
                    while (rc < right_NC) {
                        const RightIndex_ rc_end = rc + sanisizer::min(options.secondary_block_size, right_NC - rc);

                        for (LeftIndex_ cd_counter = 0; cd_counter < cd_num; ++cd_counter) {
                            const auto leftcol = left_ptrs[cd_counter];
                            const auto rightrow = right_ptrs[cd_counter];
                            for (auto lr_copy = lr; lr_copy < lr_end; ++lr_copy) {
                                const Output_ mult = leftcol[lr_copy];
                                for (auto rc_copy = rc; rc_copy < rc_end; ++rc_copy) {
                                    outptr[sanisizer::nd_offset<std::size_t>(rc_copy, right_NC, lr_copy)] += mult * static_cast<Output_>(rightrow[rc_copy]);
                                }
                            }
                        }

                        rc = rc_end;
                    }
                    lr = lr_end;
                }
                cd += cd_num;
            }
        }

        if (do_parallel && t > 0) {
            (*tmp_results)[t - 1] = std::move(tmp_output);
        }
    }, common_dim, options.num_threads);

    if (do_parallel) {
        for (int u = 1; u < num_used; ++u) {
            const auto& tmp = *((*tmp_results)[u - 1]);
            const auto N = tmp.size();
            for (I<decltype(N)> x = 0; x < N; ++x) {
                output[x] += tmp[x];
            }
        }
    }
}

}

#endif
