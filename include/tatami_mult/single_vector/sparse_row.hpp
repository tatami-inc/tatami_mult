#ifndef TATAMI_MULT_SINGLE_VECTOR_SPARSE_ROW_HPP
#define TATAMI_MULT_SINGLE_VECTOR_SPARSE_ROW_HPP

#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"

#include "../sparse_dot_product.hpp"

/**
 * @file sparse_row.hpp
 * @brief Sparse row-major LHS, single vector RHS.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_sparse_row_with_single_vector()`.
 */
struct MultiplySparseRowWithSingleVectorOptions {
    /**
     * Number of threads to use.
     */
    int num_threads = 1;
};

/**
 * @tparam accumulators_ Number of accumulators for computing the dot product,
 * see the @ref multiple-accumulators "Multiple accumulators" section for more details.
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the matrix index.
 * @tparam Right_ Numeric type of the vector on the right hand side.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer row access, but will work with all matrices.
 * @param[in] right Pointer to an array of length equal to the number of columns of `left`,
 * containing the RHS vector.
 * @param[out] output Pointer to an array of length equal to the number of rows of `left`.
 * On output, this stores the product `left * right`.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename Value_, typename Index_, typename Right_, typename Output_>
void multiply_sparse_row_with_single_vector(
    const tatami::Matrix<Value_, Index_>& left,
    const Right_* const right,
    Output_* const output,
    const MultiplySparseRowWithSingleVectorOptions& options
) {
    const Index_ NR = left.nrow();
    const Index_ NC = left.ncol();
    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(left, true, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);
        for (Index_ r = start, end = start + length; r < end; ++r) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());
            // tatami guarantees that range.number will fit in a std::size_t, so no need to protect the function call.
            output[r] = sparse_dot_product<accumulators_>(range.number, range.value, range.index, right, static_cast<Output_>(0));
        }
    }, NR, options.num_threads);
}

}

#endif
