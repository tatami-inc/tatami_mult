#ifndef TATAMI_MULT_DENSE_MATRIX_DENSE_COLUMN_DISPATCH_HPP
#define TATAMI_MULT_DENSE_MATRIX_DENSE_COLUMN_DISPATCH_HPP

#include "row_to_row.hpp"
#include "row_to_column.hpp"
#include "column_to_row.hpp"
#include "column_to_column.hpp"

#include "../../utils.hpp"

/**
 * @file dispatch.hpp
 * @brief Dense column LHS, dense matrix RHS
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_dense_column_with_dense_matrix()`.
 */
struct MultiplyDenseColumnWithDenseMatrixOptions {
    /**
     * Options to pass to `multiply_dense_column_with_dense_column_matrix_to_column_output()`,
     * if `right` is a column-major matrix and `output_row_major == false`.
     */
    MultiplyDenseColumnWithDenseColumnMatrixToColumnOutputOptions column_to_column;

    /**
     * Options to pass to `multiply_dense_column_with_dense_column_matrix_to_row_output()`,
     * if `right` is a column-major matrix and `output_row_major == true`.
     */
    MultiplyDenseColumnWithDenseColumnMatrixToRowOutputOptions column_to_row;

    /**
     * Options to pass to `multiply_dense_column_with_dense_column_matrix_to_column_output()`,
     * if `right` is a row-major matrix and `output_row_major == false`.
     */
    MultiplyDenseColumnWithDenseRowMatrixToColumnOutputOptions row_to_column;

    /**
     * Options to pass to `multiply_dense_column_with_dense_column_matrix_to_row_output()`,
     * if `right` is a row-major matrix and `output_row_major == true`.
     */
    MultiplyDenseColumnWithDenseRowMatrixToRowOutputOptions row_to_row;
};

/**
 * @tparam LeftValue_ Numeric type of the left matrix value.
 * @tparam LeftIndex_ Integer type of the left matrix index.
 * @tparam RightValue_ Numeric type of the right matrix value.
 * @tparam RightIndex_ Integer type of the right matrix index.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer column access, but will work with all matrices.
 * @param right RHS matrix to be multiplied.
 * The number of rows in this matrix should be equal to the number of columns in `left`.
 * This function is optimized for dense matrices, but will work with all matrices.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right` in either row- or column-major format depending on `output_row_major`.
 * @param output_row_major Whether to store the matrix product in row-major format in `output`.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_dense_column_with_dense_matrix(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const bool output_row_major,
    const MultiplyDenseColumnWithDenseMatrixOptions& options
) {
    if (right.prefer_rows()) {
        if (output_row_major) {
            multiply_dense_column_with_dense_row_matrix_to_row_output(left, right, output, options.row_to_row);
        } else {
            multiply_dense_column_with_dense_row_matrix_to_column_output(left, right, output, options.row_to_column);
        }

    } else {
        if (output_row_major) {
            multiply_dense_column_with_dense_column_matrix_to_row_output(left, right, output, options.column_to_row);
        } else {
            multiply_dense_column_with_dense_column_matrix_to_column_output(left, right, output, options.column_to_column);
        }
    }
}

}

#endif
