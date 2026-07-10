#ifndef TATAMI_MULT_DENSE_MATRIX_DENSE_ROW_DISPATCH_HPP
#define TATAMI_MULT_DENSE_MATRIX_DENSE_ROW_DISPATCH_HPP

#include "row_to_row.hpp"
#include "row_to_column.hpp"
#include "column_to_row.hpp"
#include "column_to_column.hpp"

#include "../../utils.hpp"

namespace tatami_mult {

/**
 * @brief Options for `multiply_dense_row_with_dense_matrix()`.
 */
struct MultiplyDenseRowWithDenseMatrixOptions {
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
 * The number of rows in this matrix should be equal to the number of columns in `left`.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right`.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_dense_row_with_dense_matrix(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    const bool output_row_major,
    Output_* const output,
    const MultiplyDenseRowWithDenseMatrixOptions& options
) {
    const auto rNR = right.nrow();
    const auto rNC = right.ncol();

    if (right.prefer_rows()) {
        auto all_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(rNR);
        auto all_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(rNR);
        populate_dense_buffers(true, rNR, rNC, right, all_buffers, all_ptrs, options.num_threads);

        if (output_row_major) {
            MultiplyDenseRowWithDenseRowMatrixToRowOutputOptions opt;
            opt.num_threads = options.num_threads;
            opt.primary_block_size = options.primary_block_size;
            opt.secondary_block_size = options.secondary_block_size;
            multiply_dense_row_with_dense_row_matrix_to_row_output(left, all_ptrs, rNC, output, opt);
        } else {
            MultiplyDenseRowWithDenseRowMatrixToColumnOutputOptions opt;
            opt.num_threads = options.num_threads;
            opt.primary_block_size = options.primary_block_size;
            opt.secondary_block_size = options.secondary_block_size;
            multiply_dense_row_with_dense_row_matrix_to_column_output(left, all_ptrs, rNC, output, opt);
        }

    } else {
        auto all_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(rNC);
        auto all_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(rNC);
        populate_dense_buffers(false, rNC, rNR, right, all_buffers, all_ptrs, options.num_threads);

        if (output_row_major) {
            MultiplyDenseRowWithDenseColumnMatrixToRowOutputOptions opt;
            opt.num_threads = options.num_threads;
            opt.primary_block_size = options.primary_block_size;
            opt.secondary_block_size = options.secondary_block_size;
            multiply_dense_row_with_dense_column_matrix_to_row_output(left, all_ptrs, rNC, output, opt);
        } else {
            MultiplyDenseRowWithDenseColumnMatrixToColumnOutputOptions opt;
            opt.num_threads = options.num_threads;
            opt.primary_block_size = options.primary_block_size;
            opt.secondary_block_size = options.secondary_block_size;
            multiply_dense_row_with_dense_column_matrix_to_column_output(left, all_ptrs, rNC, output, opt);
        }
    }
}

}

#endif
