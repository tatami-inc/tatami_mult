#ifndef TATAMI_MULT_DENSE_MATRIX_SPARSE_ROW_ROW_TO_COLUMN_HPP
#define TATAMI_MULT_DENSE_MATRIX_SPARSE_ROW_ROW_TO_COLUMN_HPP

#include <cstddef>
#include <vector>
#include <algorithm>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../../utils.hpp"

/**
 * @file row_to_row.hpp
 * @brief Sparse row LHS, dense row-major RHS, column-major output.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_sparse_row_with_dense_row_matrix_to_column_output()`.
 */
struct MultiplySparseRowWithDenseRowMatrixToColumnOutputOptions {
    /**
     * Number of threads to use.
     */
    int num_threads = 1;
};

/**
 * @tparam LeftValue_ Numeric type of the left matrix value.
 * @tparam LeftIndex_ Integer type of the left matrix index.
 * @tparam RightValue_ Numeric type of the right matrix value.
 * @tparam RightIndex_ Integer type of the right matrix index.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer row access, but will work with all matrices.
 * @param right RHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * The number of rows in this matrix should be equal to the number of columns in `left`.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right` in column-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_sparse_row_with_dense_row_matrix_to_column_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplySparseRowWithDenseRowMatrixToColumnOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right.ncol();

    auto right_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(common_dim);
    auto right_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(common_dim);
    populate_dense_buffers(true, common_dim, right_NC, right, right_buffers, right_ptrs, options.num_threads);

    tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(left, true, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(common_dim);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<LeftIndex_> >(common_dim);
        auto tmp_output = tatami::create_container_of_Index_size<std::vector<Output_> >(right_NC);

        for (LeftIndex_ lr = 0; lr < length; ++lr) {
            const auto range = ext->fetch(vbuffer.data(), ibuffer.data());

            for (LeftIndex_ x = 0; x < range.number; ++x) {
                const auto rightrow = right_ptrs[range.index[x]];
                const auto mult = range.value[x];
                for (RightIndex_ rc = 0; rc < right_NC; ++rc) {
                    tmp_output[rc] += mult * rightrow[rc];
                }
            }

            // Transposing to the output once all summations are done.
            for (RightIndex_ rc = 0; rc < right_NC; ++rc) {
                output[sanisizer::nd_offset<std::size_t>(start + lr, left_NR, rc)] = tmp_output[rc];
                tmp_output[rc] = 0;
            }
        }
    }, left_NR, options.num_threads);
}

}

#endif
