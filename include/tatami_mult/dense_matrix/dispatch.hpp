#ifndef TATAMI_MULT_DENSE_MATRIX_DISPATCH_HPP
#define TATAMI_MULT_DENSE_MATRIX_DISPATCH_HPP

#include "dense_row/dispatch.hpp"
#include "dense_column/dispatch.hpp"
#include "sparse_row/dispatch.hpp"
#include "sparse_column/dispatch.hpp"

/**
 * @file dispatch.hpp
 * @brief Any matrix LHS, dense matrix RHS.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_with_dense_matrix()`.
 */
struct MultiplyWithDenseMatrixOptions {
    /**
     * Options to pass to `multiply_dense_row_with_dense_matrix()`, if `left` is a dense matrix that prefers row access.
     */
    MultiplyDenseRowWithDenseMatrixOptions dense_row;

    /**
     * Options to pass to `multiply_dense_column_with_dense_matrix()`, if `left` is a dense matrix that prefers column access.
     */
    MultiplyDenseColumnWithDenseMatrixOptions dense_column;

    /**
     * Options to pass to `multiply_sparse_row_with_dense_matrix()`, if `left` is a sparse matrix that prefers row access.
     */
    MultiplySparseRowWithDenseMatrixOptions sparse_row;

    /**
     * Options to pass to `multiply_sparse_column_with_dense_matrix()`, if `left` is a sparse matrix that prefers column access.
     */
    MultiplySparseColumnWithDenseMatrixOptions sparse_column;
};

/**
 * @tparam LeftValue_ Numeric type of the left matrix value.
 * @tparam LeftIndex_ Integer type of the left matrix index.
 * @tparam RightValue_ Numeric type of the right matrix value.
 * @tparam RightIndex_ Integer type of the right matrix index.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * @param right RHS matrix to be multiplied.
 * The number of rows in this matrix should be equal to the number of columns in `left`.
 * This function is optimized for dense matrices, but will work with all matrices.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right` in either row- or column-major format depending on `output_row_major`.
 * @param output_row_major Whether to store the matrix product in row-major format in `output`.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_with_dense_matrix(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const bool output_row_major,
    const MultiplyWithDenseMatrixOptions& options
) {
    if (left.is_sparse()) {
        if (left.prefer_rows()) {
            multiply_sparse_row_with_dense_matrix(left, right, output, output_row_major, options.sparse_row);
        } else {
            multiply_sparse_column_with_dense_matrix(left, right, output, output_row_major, options.sparse_column);
        }
    } else {
        if (left.prefer_rows()) {
            multiply_dense_row_with_dense_matrix(left, right, output, output_row_major, options.dense_row);
        } else {
            multiply_dense_column_with_dense_matrix(left, right, output, output_row_major, options.dense_column);
        }
    }
}

}

#endif
