#ifndef TATAMI_MULT_MULTIPLE_VECTORS_DENSE_COLUMN_PUBLIC_HPP
#define TATAMI_MULT_MULTIPLE_VECTORS_DENSE_COLUMN_PUBLIC_HPP

#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"

#include "internal.hpp"

/**
 * @file public.hpp
 * @brief Dense column LHS, multiple vectors RHS.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_dense_column_with_multiple_vectors()`.
 */
struct MultiplyDenseColumnWithMultipleVectorsOptions {
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
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the matrix index.
 * @tparam Right_ Numeric type of the vector on the right hand side.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer column access, but will work with all matrices.
 * @param[in] right Vector of pointers, each of which points to an array of length `left.ncol()`.
 * Each entry contains a vector with which to multiply `left`.
 * @param[out] output Vector of pointers, each of which points to an array of length `left.nrow()`.
 * On output, the `i`-th entry stores the product `left * right[i]`.
 * @param options Further options.
 */
template<typename Value_, typename Index_, typename Right_, typename Output_>
void multiply_dense_column_with_multiple_vectors(
    const tatami::Matrix<Value_, Index_>& left,
    const std::vector<Right_*>& right,
    const std::vector<Output_*>& output,
    const MultiplyDenseColumnWithMultipleVectorsOptions& options
) {
    multiply_dense_column_with_some_vectors(
        left,
        right,
        [&](const I<decltype(right.size())> h) -> Output_* {
            return output[h];
        },
        [&](){
            MultiplyDenseColumnWithSomeVectorsOptions opt;
            opt.num_threads = options.num_threads;
            opt.primary_block_size = options.primary_block_size;
            opt.secondary_block_size = options.secondary_block_size;
            return opt;
        }()
    );
}

}

#endif
