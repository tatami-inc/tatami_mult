#ifndef TATAMI_MULT_SPARSE_MATRIX_DENSE_ROW_DISPATCH_HPP
#define TATAMI_MULT_SPARSE_MATRIX_DENSE_ROW_DISPATCH_HPP

#include "row_to_row.hpp"
#include "row_to_column.hpp"
#include "column_to_row.hpp"
#include "column_to_column.hpp"

#include "../../utils.hpp"

/**
 * @file dispatch.hpp
 * @brief Dense row-major LHS, dense matrix RHS.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_dense_row_with_sparse_matrix()`.
 */
struct MultiplyDenseRowWithSparseMatrixOptions {
    /**
     * Options to pass to `multiply_dense_row_with_dense_column_matrix_to_column_output()`,
     * if `right` is a column-major matrix and `output_row_major == false`.
     */
    MultiplyDenseRowWithSparseColumnMatrixToColumnOutputOptions column_to_column;

    /**
     * Options to pass to `multiply_dense_row_with_dense_column_matrix_to_row_output()`,
     * if `right` is a column-major matrix and `output_row_major == true`.
     */
    MultiplyDenseRowWithSparseColumnMatrixToRowOutputOptions column_to_row;

    /**
     * Options to pass to `multiply_dense_row_with_dense_row_matrix_to_column_output()`,
     * if `right` is a row-major matrix and `output_row_major == false`.
     */
    MultiplyDenseRowWithSparseRowMatrixToColumnOutputOptions row_to_column;

    /**
     * Options to pass to `multiply_dense_row_with_dense_row_matrix_to_row_output()`,
     * if `right` is a row-major matrix and `output_row_major == true`.
     */
    MultiplyDenseRowWithSparseRowMatrixToRowOutputOptions row_to_row;
};

/**
 * Set the number of threads to use in all multiplication functions involving a dense row-major LHS and a dense matrix RHS.
 *
 * @param options Options to be set.
 * @param num_threads Number of threads, should be positive.
 */
inline void set_num_threads(MultiplyDenseRowWithSparseMatrixOptions& options, int num_threads) {
    options.column_to_column.num_threads = num_threads;
    options.column_to_row.num_threads = num_threads;
    options.row_to_column.num_threads = num_threads;
    options.row_to_row.num_threads = num_threads;
}

/**
 * Set the block size to use in all multiplication functions involving a dense row-major LHS and a sparse matrix RHS.
 *
 * @param options Options to be set.
 * @param block_size Block size.
 */
inline void set_sparse_block_size(MultiplyDenseRowWithSparseMatrixOptions& options, int block_size) {
    options.column_to_column.block_size = block_size;
    options.column_to_row.block_size = block_size;
    options.row_to_column.block_size = block_size;
    options.row_to_row.block_size = block_size;
}

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
 * This function is optimized for sparse matrices, but will work with all matrices.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right` in either row- or column-major format depending on `output_row_major`.
 * @param output_row_major Whether to store the matrix product in row-major format in `output`.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_dense_row_with_sparse_matrix(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const bool output_row_major,
    const MultiplyDenseRowWithSparseMatrixOptions& options
) {
    if (right.prefer_rows()) {
        if (output_row_major) {
            multiply_dense_row_with_sparse_row_matrix_to_row_output(left, right, output, options.row_to_row);
        } else {
            multiply_dense_row_with_sparse_row_matrix_to_column_output(left, right, output, options.row_to_column);
        }

    } else {
        if (output_row_major) {
            multiply_dense_row_with_sparse_column_matrix_to_row_output(left, right, output, options.column_to_row);
        } else {
            multiply_dense_row_with_sparse_column_matrix_to_column_output(left, right, output, options.column_to_column);
        }
    }
}

}

#endif
