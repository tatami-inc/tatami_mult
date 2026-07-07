#ifndef TATAMI_MULT_SINGLE_VECTOR_DISPATCH_HPP
#define TATAMI_MULT_SINGLE_VECTOR_DISPATCH_HPP

#include "dense_row.hpp"
#include "dense_column.hpp"
#include "sparse_row.hpp"
#include "sparse_column.hpp"

/**
 * @file dispatch.hpp
 * @brief Any matrix LHS, single vector RHS.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_with_single_vector()`.
 */
struct MultiplyWithSingleVectorOptions {
    /**
     * Number of threads to use.
     */
    int num_threads = 1;
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
 * @param[in] right Pointer to an array of length equal to the number of columns of `left`,
 * containing the RHS vector.
 * @param[out] output Pointer to an array of length equal to the number of rows of `left`.
 * On output, this stores the product `left * right`.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename Value_, typename Index_, typename Right_, typename Output_>
void multiply_with_single_vector(
    const tatami::Matrix<Value_, Index_>& left,
    const Right_* const right,
    Output_* const output,
    const MultiplyWithSingleVectorOptions& options
) {
    if (left.is_sparse()) {
        if (left.prefer_rows()) {
            multiply_sparse_row_with_single_vector<accumulators_>(left, right, output, [&](){
                MultiplySparseRowWithSingleVector opt;
                opt.num_threads = options.num_threads;
                return opt;
            }());
        } else {
            multiply_sparse_column_with_single_vector(left, right, output, [&](){
                MultiplySparseColumnWithSingleVector opt;
                opt.num_threads = options.num_threads;
                return opt;
            }());
        }

    } else {
        if (left.prefer_rows()) {
            multiply_dense_row_with_single_vector<accumulators_>(left, right, output, [&](){
                MultiplyDenseRowWithSingleVector opt;
                opt.num_threads = options.num_threads;
                return opt;
            }());
        } else {
            multiply_dense_column_with_single_vector(left, right, output, [&](){
                MultiplyDenseColumnWithSingleVector opt;
                opt.num_threads = options.num_threads;
                return opt;
            }());
        }
    }
}

}

#endif
