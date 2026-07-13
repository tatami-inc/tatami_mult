#ifndef TATAMI_MULT_MULTIPLE_VECTORS_DISPATCH_HPP
#define TATAMI_MULT_MULTIPLE_VECTORS_DISPATCH_HPP

#include "dense_row.hpp"
#include "dense_column.hpp"
#include "sparse_row.hpp"
#include "sparse_column.hpp"

/**
 * @file dispatch.hpp
 * @brief Any matrix LHS, multiple vectors RHS.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_with_multiple_vectors()`.
 */
struct MultiplyWithMultipleVectorsOptions {
    /**
     * Options to pass to `multiply_dense_row_with_multiple_vectors()`, if `left` is a dense matrix that prefers row access.
     */
    MultiplyDenseRowWithMultipleVectorsOptions dense_row;

    /**
     * Options to pass to `multiply_dense_column_with_multiple_vectors()`, if `left` is a dense matrix that prefers column access.
     */
    MultiplyDenseColumnWithMultipleVectorsOptions dense_column;

    /**
     * Options to pass to `multiply_sparse_row_with_multiple_vectors()`, if `left` is a sparse matrix that prefers row access.
     */
    MultiplySparseRowWithMultipleVectorsOptions sparse_row;

    /**
     * Options to pass to `multiply_sparse_column_with_multiple_vectors()`, if `left` is a sparse matrix that prefers column access.
     */
    MultiplySparseColumnWithMultipleVectorsOptions sparse_column;
};

/**
 * Set the number of threads to use in all multiplication functions involving multiple vectors RHS.
 *
 * @param options Options to be set.
 * @param num_threads Number of threads, should be positive.
 */
inline void set_num_threads(MultiplyWithMultipleVectorsOptions& options, int num_threads) {
    options.dense_row.num_threads = num_threads;
    options.dense_column.num_threads = num_threads;
    options.sparse_row.num_threads = num_threads;
    options.sparse_column.num_threads = num_threads;
}

/**
 * Set the primary block size to use in all multiplication functions involving a dense matrix LHS and multiple vectors RHS.
 *
 * @param options Options to be set.
 * @param primary_block_size Primary block size.
 */
inline void set_dense_primary_block_size(MultiplyWithMultipleVectorsOptions& options, int primary_block_size) {
    options.dense_row.primary_block_size = primary_block_size;
    options.dense_column.primary_block_size = primary_block_size;
}

/**
 * Set the secondary block size to use in all multiplication functions involving a dense matrix LHS and multiple vectors RHS.
 *
 * @param options Options to be set.
 * @param secondary_block_size Secondary block size.
 */
inline void set_dense_secondary_block_size(MultiplyWithMultipleVectorsOptions& options, int secondary_block_size) {
    options.dense_row.secondary_block_size = secondary_block_size;
    options.dense_column.secondary_block_size = secondary_block_size;
}

/**
 * Set the block size to use in all multiplication functions involving a sparse matrix LHS and multiple vectors RHS.
 *
 * @param options Options to be set.
 * @param block_size Block size.
 */
inline void set_sparse_block_size(MultiplyWithMultipleVectorsOptions& options, int block_size) {
    options.sparse_row.block_size = block_size;
    options.sparse_column.block_size = block_size;
}

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
 * @param[in] right Vector of pointers, each of which points to an array of length `left.ncol()`.
 * Each entry contains a vector with which to multiply `left`.
 * @param[out] output Vector of pointers, each of which points to an array of length `left.nrow()`.
 * On output, the `i`-th entry stores the product `left * right[i]`.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename Value_, typename Index_, typename Right_, typename Output_>
void multiply_with_multiple_vectors(
    const tatami::Matrix<Value_, Index_>& left,
    const std::vector<Right_*>& right,
    const std::vector<Output_*>& output,
    const MultiplyWithMultipleVectorsOptions& options
) {
    if (left.is_sparse()) {
        if (left.prefer_rows()) {
            multiply_sparse_row_with_multiple_vectors<accumulators_>(left, right, output, options.sparse_row);
        } else {
            multiply_sparse_column_with_multiple_vectors(left, right, output, options.sparse_column);
        }
    } else {
        if (left.prefer_rows()) {
            multiply_dense_row_with_multiple_vectors<accumulators_>(left, right, output, options.dense_row);
        } else {
            multiply_dense_column_with_multiple_vectors(left, right, output, options.dense_column);
        }
    }
}

}

#endif
