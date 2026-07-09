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
            multiply_sparse_row_with_multiple_vectors<accumulators_>(left, right, output, [&](){
                MultiplySparseRowWithMultipleVectorsOptions opt;
                opt.num_threads = options.num_threads;
                opt.block_size = options.primary_block_size;
                return opt;
            }());
        } else {
            multiply_sparse_column_with_multiple_vectors(left, right, output, [&](){
                MultiplySparseColumnWithMultipleVectorsOptions opt;
                opt.num_threads = options.num_threads;
                opt.block_size = options.primary_block_size;
                return opt;
            }());
        }

    } else {
        if (left.prefer_rows()) {
            multiply_dense_row_with_multiple_vectors<accumulators_>(left, right, output, [&](){
                MultiplyDenseRowWithMultipleVectorsOptions opt;
                opt.num_threads = options.num_threads;
                opt.primary_block_size = options.primary_block_size;
                opt.secondary_block_size = options.secondary_block_size;
                return opt;
            }());
        } else {
            multiply_dense_column_with_multiple_vectors(left, right, output, [&](){
                MultiplyDenseColumnWithMultipleVectorsOptions opt;
                opt.num_threads = options.num_threads;
                opt.primary_block_size = options.primary_block_size;
                opt.secondary_block_size = options.secondary_block_size;
                return opt;
            }());
        }
    }
}

}

#endif
