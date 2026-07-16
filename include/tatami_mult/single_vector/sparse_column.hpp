#ifndef TATAMI_MULT_SPARSE_COLUMN_HPP
#define TATAMI_MULT_SPARSE_COLUMN_HPP

#include <vector>
#include <optional>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../utils.hpp"

/**
 * @file sparse_column.hpp
 * @brief Sparse column-major LHS, single vector RHS.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_sparse_column_with_single_vector()`.
 */
struct MultiplySparseColumnWithSingleVectorOptions {
    /**
     * Number of threads to use.
     */
    int num_threads = 1;
};

/**
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the matrix index.
 * @tparam Right_ Numeric type of the vector on the right hand side.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer column access, but will work with all matrices.
 * @param[in] right Pointer to an array of length equal to the number of columns of `left`,
 * containing the RHS vector.
 * @param[out] output Pointer to an array of length equal to the number of rows of `left`.
 * On output, this stores the product `left * right`.
 * @param options Further options.
 */
template<typename Value_, typename Index_, typename Right_, typename Output_>
void multiply_sparse_column_with_single_vector(
    const tatami::Matrix<Value_, Index_>& left,
    const Right_* const rhs,
    Output_* const output,
    const MultiplySparseColumnWithSingleVectorOptions& options
) {
    const Index_ NR = left.nrow();
    const Index_ NC = left.ncol();

    const bool do_parallel = options.num_threads > 1; 
    std::optional<std::vector<std::optional<std::vector<Output_> > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(options.num_threads - 1));
    }
    std::fill_n(output, NR, 0);

    const auto num_used = tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(left, false, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NR);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NR);

        Output_* optr;
        std::optional<std::vector<Output_> > cur_output;
        if (!do_parallel || t == 0) {
            optr = output;
        } else {
            cur_output.emplace(tatami::cast_Index_to_container_size<I<decltype(*cur_output)> >(NR));
            optr = cur_output->data();
        }

        for (Index_ c = 0; c < length; ++c) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());
            const Output_ mult = rhs[start + c];
            for (Index_ r = 0; r < range.number; ++r) {
                optr[range.index[r]] += mult * range.value[r];
            }
        }

        if (do_parallel && t > 0) {
            (*tmp_results)[t - 1] = std::move(cur_output);            
        }
    }, NC, options.num_threads);

    if (do_parallel) {
        for (int u = 1; u < num_used; ++u) {
            const auto& tmp = *((*tmp_results)[u - 1]);
            for (Index_ r = 0; r < NR; ++r) {
                output[r] += tmp[r];
            }
        }
    }
}

}

#endif
