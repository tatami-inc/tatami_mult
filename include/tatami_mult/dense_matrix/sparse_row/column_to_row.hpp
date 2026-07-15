#ifndef TATAMI_MULT_DENSE_MATRIX_SPARSE_ROW_COLUMN_TO_ROW_HPP
#define TATAMI_MULT_DENSE_MATRIX_SPARSE_ROW_COLUMN_TO_ROW_HPP

#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"

#include "../../utils.hpp"
#include "../../sparse_dot_product.hpp"

/**
 * @file column_to_row.hpp
 * @brief Sparse row-major LHS, dense column-major RHS, row-major output.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_sparse_row_with_dense_column_matrix_to_row_output()`.
 */
struct MultiplySparseRowWithDenseColumnMatrixToRowOutputOptions {
    /**
     * Number of threads to use.
     */
    int num_threads = 1;

    /**
     * Block size, i.e., the number of LHS rows to be loaded at once.
     * See the \f$B\f$ parameter in the @ref sparse-blocking "Blocking for sparse matrices" section for more details.
     */
    int block_size = 16;
};

/**
 * @tparam accumulators_ Number of accumulators for computing the dot product,
 * see the @ref multiple-accumulators "Multiple accumulators" section for more details.
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
 * On output, this stores the product of `left` and `right` in row-major format.
 * @param options Further options.
 */
template<std::size_t accumulators_, typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_sparse_row_with_dense_column_matrix_to_row_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplySparseRowWithDenseColumnMatrixToRowOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right.ncol();

    auto right_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(right_NC);
    auto right_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(right_NC);
    populate_dense_buffers(false, right_NC, common_dim, right, right_buffers, right_ptrs, options.num_threads);

    if (options.block_size == 1) {
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto ext = tatami::consecutive_extractor<true>(left, true, start, length);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(common_dim);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<LeftIndex_> >(common_dim);

            for (LeftIndex_ lr = 0; lr < length; ++lr) {
                const auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                for (RightIndex_ rc = 0; rc < right_NC; ++rc) {
                    // Implicit cast of range.number to size_t is safe, as per the tatami contract.
                    // Also some false sharing potential here, but we just touch each location once per outer loop, so it's fine.
                    const auto val = sparse_dot_product<accumulators_>(range.number, range.value, range.index, right_ptrs[rc], static_cast<Output_>(0));
                    output[sanisizer::nd_offset<std::size_t>(rc, right_NC, start + lr)] = val;
                }
            }
        }, left_NR, options.num_threads);
        return;
    }

    tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(left, true, start, length);

        // Our blocking strategy is to collect multiple LHS rows so that, for each RHS vector,
        // we can keep it in cache for easy re-use when computing the dot-product for each LHS row.
        std::vector<std::vector<LeftValue_> > left_vbuffers;
        std::vector<std::vector<LeftIndex_> > left_ibuffers;
        std::vector<tatami::SparseRange<LeftValue_, LeftIndex_> > left_ranges;
        std::vector<LeftIndex_> left_non_empty;
        {
            const LeftIndex_ max_block_rows = sanisizer::min(length, options.block_size);
            left_vbuffers.reserve(max_block_rows);
            left_ibuffers.reserve(max_block_rows);
            for (LeftIndex_ lr = 0; lr < max_block_rows; ++lr) {
                left_vbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(common_dim));
                left_ibuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftIndex_> >(common_dim));
            }
            sanisizer::resize(left_ranges, max_block_rows);
            left_non_empty.reserve(max_block_rows);
        }

        LeftIndex_ lr = 0;
        while (lr < length) {
            // We only consider the LHS rows with at least one structural non-zero.
            // Thus, our block consists of 'options.block_size' non-empty LHS rows, rather than fixed row-wise chunks of the LHS matrix.
            // This ensures that we don't waste iterations on LHS rows that will only have zeros in the output matrix (and are filled as such).
            const auto left_block_info = fetch_non_empty_sparse_block(
                *ext,
                left_vbuffers,
                left_ibuffers,
                left_ranges,
                left_non_empty,
                lr,
                length,
                options.block_size,
                /* zero = */ [&](const LeftIndex_ lr_copy) -> void {
                    std::fill_n(output + sanisizer::product_unsafe<std::size_t>(start + lr_copy, right_NC), right_NC, 0);
                }
            );
            const auto lr_num = left_block_info.num_non_empty;

            // If the LHS columns are all non-empty, we can speed up the loops by just using a simple counter to get the column indices.
            // Otherwise, we'll have to access the 'left_non_empty' vector to figure out the indices of each non-empty column.
            if (left_block_info.all_non_empty) {
                const auto lr_base = lr + start; 
                // Yes, we deliberately iterate over the RHS columns in the outer loop to keep the dense column in cache across multiple LHS vectors.
                // If we did it the other way around, this would defeat the purpose of blocking.
                for (RightIndex_ rc = 0; rc < right_NC; ++rc) {
                    const auto rcol = right_ptrs[rc];
                    for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                        const auto& currange = left_ranges[lr_counter];
                        output[sanisizer::nd_offset<std::size_t>(rc, right_NC, lr_base + lr_counter)] = sparse_dot_product<accumulators_>(
                            currange.number, // Implicit cast of range.number to size_t is safe, as per the tatami contract.
                            currange.value,
                            currange.index,
                            rcol,
                            static_cast<Output_>(0)
                        );
                    }
                }

            } else {
                for (auto& lrne : left_non_empty) {
                    lrne += start;
                }
                for (RightIndex_ rc = 0; rc < right_NC; ++rc) { // again, iterating over the RHS columns in the outer loop, see above.
                    const auto rcol = right_ptrs[rc];
                    for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                        const auto& currange = left_ranges[lr_counter];
                        output[sanisizer::nd_offset<std::size_t>(rc, right_NC, left_non_empty[lr_counter])] = sparse_dot_product<accumulators_>(
                            currange.number, // see above.
                            currange.value,
                            currange.index,
                            rcol,
                            static_cast<Output_>(0)
                        );
                    }
                }
            }

            lr = left_block_info.position;
        }
    }, left_NR, options.num_threads);
}

}

#endif
