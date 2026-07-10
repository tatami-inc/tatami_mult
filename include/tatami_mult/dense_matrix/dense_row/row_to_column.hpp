#ifndef TATAMI_MULT_DENSE_MATRIX_DENSE_ROW_ROW_TO_COLUMN_HPP
#define TATAMI_MULT_DENSE_MATRIX_DENSE_ROW_ROW_TO_COLUMN_HPP

#include <vector>
#include <cstddef>

#include "tatami/tatami.hpp"

/**
 * @file row_to_column.hpp
 * @brief Dense row LHS, dense row-major matrix RHS, column-major output.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_dense_row_with_dense_row_matrix_to_column_output()`.
 */
struct MultiplyDenseRowWithDenseRowMatrixToColumnOutputOptions {
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
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * The number of rows in this matrix should be equal to the number of columns in `left`.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right` in column-major format.
 * @param options Further options.
 */
template< typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_dense_row_with_dense_row_matrix_to_column_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const std::vector<const RightValue_*>& right,
    [[maybe_unused]] const RightIndex_ num_rhs,
    Output_* const output,
    const MultiplyDenseRowWithDenseRowMatrixToColumnOutputOptions& options
) {
    const auto NR = left.nrow();
    const auto NC = left.ncol();

    if (options.primary_block_size == 1) {
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto ext = tatami::consecutive_extractor<false>(left, true, start, length);
            auto buffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(NC);

            // Use a temporary buffer to mimic an output row.
            // This gives us contiguous writes in the innermost loop while mitigating false sharing.
            auto tmp_output = tatami::create_container_of_Index_size<std::vector<Output_> >(num_rhs);

            for (LeftIndex_ r = 0; r < length; ++r) {
                const auto left_ptr = ext->fetch(buffer.data());
                std::fill(tmp_output.begin(), tmp_output.end(), 0);
                for (LeftIndex_ c = 0; c < NC; ++c) {
                    const Output_ mult = left_ptr[c];
                    const auto rightrow = right[c];
                    for (RightIndex_ h = 0; h < num_rhs; ++h) {
                        tmp_output[h] += static_cast<Output_>(rightrow[h]) * mult;
                    }
                }
                for (RightIndex_ h = 0; h < num_rhs; ++h) {
                    output[sanisizer::nd_offset<std::size_t>(start + r, NR, h)] = tmp_output[h];
                }
            }
        }, NR, options.num_threads);

    } else {
        const auto primary_block_size = sanisizer::cast<LeftIndex_>(options.primary_block_size);
        const auto secondary_block_size = sanisizer::cast<RightIndex_>(options.secondary_block_size);
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto left_ext = tatami::consecutive_extractor<false>(left, true, start, length);
            std::vector<std::vector<LeftValue_> > left_buffers;
            std::vector<const LeftValue_*> left_ptrs;

            std::vector<Output_> tmp_output;
            {
                const LeftIndex_ max_block_rows = std::min(length, primary_block_size);
                left_buffers.reserve(max_block_rows);
                for (LeftIndex_  b = 0; b < max_block_rows; ++b) {
                    left_buffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(NC));
                }
                sanisizer::resize(left_ptrs, max_block_rows);

                // Creating a block to hold the output during the updates over all 'c'.
                // This enables contiguous writes in the innermost loop while also avoiding false sharing.
                // Hopefully std::vector can store this - I'd be very surprised if size_type != size_t, but we check anyway.
                sanisizer::resize(tmp_output, sanisizer::product_unsafe<std::size_t>(max_block_rows, num_rhs));
            }

            LeftIndex_ l = 0;
            while (l < length) {
                const auto lnum = std::min<LeftIndex_>(primary_block_size, length - l);
                for (LeftIndex_ lcounter = 0; lcounter < lnum; ++lcounter) {
                    left_ptrs[lcounter] = left_ext->fetch(left_buffers[lcounter].data());
                }

                const auto out_space = sanisizer::product_unsafe<std::size_t>(lnum, num_rhs);
                std::fill_n(tmp_output.data(), out_space, 0);

                LeftIndex_ c = 0;
                while (c < NC) { 
                    const LeftIndex_ cend = c + std::min<LeftIndex_>(primary_block_size, NC - c);
                    RightIndex_ h = 0;
                    while (h < num_rhs) { // jump by block to go to the right of the RHS rows as this is most contiguous.
                        const RightIndex_ hend = h + std::min<RightIndex_>(secondary_block_size, num_rhs - h);

                        for (LeftIndex_ lcounter = 0; lcounter < lnum; ++lcounter) {
                            const auto matrow = left_ptrs[lcounter];
                            const auto prod = tmp_output.data() + sanisizer::product_unsafe<std::size_t>(lcounter, num_rhs);
                            for (auto ccopy = c; ccopy < cend; ++ccopy) {
                                const auto mult = matrow[ccopy];
                                const auto& rightrow = right[ccopy];
                                for (auto hcopy = h; hcopy < hend; ++hcopy) {
                                    prod[hcopy] += mult * rightrow[hcopy];
                                }
                            }
                        }

                        h = hend;
                    }
                    c = cend;
                }

                // Doing a blocked transposition to write to the output array.
                RightIndex_ h = 0;
                while (h < num_rhs) {
                    const RightIndex_ hend = h + std::min<RightIndex_>(secondary_block_size, num_rhs - h);
                    for (LeftIndex_ lcounter = 0; lcounter < lnum; ++lcounter) {
                        for (auto hcopy = h; hcopy < hend; ++hcopy) {
                            const auto val = tmp_output[sanisizer::nd_offset<std::size_t>(hcopy, num_rhs, lcounter)];
                            output[sanisizer::nd_offset<std::size_t>(start + l + lcounter, NR, hcopy)] = val;
                        }
                    }
                    h = hend;
                }

                l += lnum;
            }
        }, NR, options.num_threads);
    }
}

}

#endif
