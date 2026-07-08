#ifndef TATAMI_MULT_MULTIPLE_VECTORS_SPARSE_ROW_PUBLIC_HPP
#define TATAMI_MULT_MULTIPLE_VECTORS_SPARSE_ROW_PUBLIC_HPP

#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"

#include "internal.hpp"
#include "../../sparse_dot_product.hpp"

/**
 * @file public.hpp
 * @brief Sparse row LHS, multiple vectors RHS.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_sparse_row_with_multiple_vectors()`.
 */
struct MultiplySparseRowWithMultipleVectorsOptions {
    /**
     * Number of threads to use.
     */
    int num_threads = 1;

    /**
     * Block size.
     */
    int block_size = 16;
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
 * This function is optimized for sparse matrices that prefer row access, but will work with all matrices.
 * @param[in] right Vector of pointers, each of which points to an array of length `left.ncol()`.
 * Each entry contains a vector with which to multiply `left`.
 * @param[out] output Vector of pointers, each of which points to an array of length `left.nrow()`.
 * On output, the `i`-th entry stores the product `left * right[i]`.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename Value_, typename Index_, typename Right_, typename Output_>
void multiply_sparse_row_with_multiple_vectors(
    const tatami::Matrix<Value_, Index_>& left,
    const std::vector<Right_*>& right,
    const std::vector<Output_*>& output,
    const MultiplySparseRowWithMultipleVectorsOptions& options
) {
    multiply_sparse_row_with_some_vectors<accumulators_>(
        left,
        right,
        [&](const I<decltype(right.size())> h) -> Output_* {
            return output[h];
        },
        [&](){
            MultiplySparseRowWithSomeVectorsOptions opt;
            opt.num_threads = options.num_threads;
            opt.block_size = options.block_size;
            return opt;
        }()
    );
}

}

#endif
