#ifndef TATAMI_MULT_DENSE_MATRIX_SPARSE_COLUMN_ROW_TO_ROW_HPP
#define TATAMI_MULT_DENSE_MATRIX_SPARSE_COLUMN_ROW_TO_ROW_HPP

#include <cstddef>
#include <vector>
#include <optional>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../../utils.hpp"

/**
 * @file row_to_row.hpp
 * @brief Sparse column-major LHS, dense row-major matrix RHS, row-major output.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_sparse_column_with_dense_row_matrix_to_row_output()`.
 */
struct MultiplySparseColumnWithDenseRowMatrixToRowOutputOptions {
    /**
     * Number of threads to use.
     */
    int num_threads = 1;
};

/**
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightValue_ Numeric type of the RHS matrix value.
 * @tparam RightIndex_ Integer type of the RHS matrix index.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer column access, but will work with all matrices.
 * @param right RHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * The number of rows in `right` should be equal to the number of columns in `left`.
 * @param[out] output Vector of pointers, each of which points to an array of length `left.nrow()`.
 * On output, this contains the product `left * right` in row-major order.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_sparse_column_with_dense_row_matrix_to_row_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplySparseColumnWithDenseRowMatrixToRowOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right.ncol();

    const bool do_parallel = options.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<Output_> > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(options.num_threads - 1));
    }

    const auto num_used = tatami::parallelize([&](int t, LeftIndex_ start, LeftIndex_ length) -> void {
        auto left_ext = tatami::consecutive_extractor<true>(left, false, start, length);
        auto right_ext = tatami::consecutive_extractor<false>(right, true, start, length);

        auto vbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(left_NR);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<LeftIndex_> >(left_NR);
        auto rbuffer = tatami::create_container_of_Index_size<std::vector<RightValue_> >(right_NC);

        std::optional<std::vector<Output_> > tmp_output;
        Output_* outptr; 
        if (!do_parallel || t == 0) {
            outptr = output;
        } else {
            tmp_output.emplace(sanisizer::product<I<decltype(tmp_output->size())> >(left_NR, right_NC));
            outptr = tmp_output->data();
        }

        for (LeftIndex_ cd = 0; cd < length; ++cd) {
            const auto lrange = left_ext->fetch(vbuffer.data(), ibuffer.data());
            const auto rptr = right_ext->fetch(rbuffer.data());
            for (LeftIndex_ x = 0; x < lrange.number; ++x) {
                const Output_ mult = lrange.value[x];
                const auto curout = outptr + sanisizer::product_unsafe<std::size_t>(lrange.index[x], right_NC);
                for (RightIndex_ rc = 0; rc < right_NC; ++rc) {
                    curout[rc] += mult * static_cast<Output_>(rptr[rc]); 
                }
            }
        }

        if (do_parallel && t > 0) {
            (*tmp_results)[t - 1] = std::move(tmp_output);
        }
    }, common_dim, options.num_threads);

    if (do_parallel) {
        for (int u = 1; u < num_used; ++u) {
            const auto& tmp = *((*tmp_results)[u - 1]);
            const auto N = tmp.size();
            for (I<decltype(N)> x = 0; x < N; ++x) {
                output[x] += tmp[x];
            }
        }
    }
}

}

#endif
