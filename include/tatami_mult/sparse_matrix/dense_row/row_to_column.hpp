#ifndef TATAMI_MULT_SPARSE_MATRIX_DENSE_ROW_ROW_TO_COLUMN_HPP
#define TATAMI_MULT_SPARSE_MATRIX_DENSE_ROW_ROW_TO_COLUMN_HPP

#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../../utils.hpp"
#include "../../sparse_dot_product.hpp"

/**
 * @file row_to_row.hpp
 * @brief Dense row-major LHS, sparse row-major RHS, row-major output.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_dense_row_with_sparse_row_matrix_to_column_output()`.
 */
struct MultiplyDenseRowWithSparseRowMatrixToColumnOutputOptions {
    /**
     * Number of threads to use.
     */
    int num_threads = 1;

    /**
     * Block size, i.e., number of LHS rows to be loaded at once.
     * We use this block to populate a contiguous array with a part of each LHS column that can be used in a fast vector multiply-add to the column-major output.
     * Larger values improve speed at the cost of increased memory usage.
     */
    int block_size = 16;
};

/**
 * @tparam LeftValue_ Numeric type of the left matrix value.
 * @tparam LeftIndex_ Integer type of the left matrix index.
 * @tparam RightValue_ Numeric type of the right matrix value.
 * @tparam RightIndex_ Integer type of the right matrix index.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * @param right RHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer column access, but will work with all matrices.
 * The number of rows in this matrix should be equal to the number of columns in `left`.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right` in row-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_dense_row_with_sparse_row_matrix_to_column_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplyDenseRowWithSparseRowMatrixToColumnOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right.ncol();

    auto right_vbuffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(common_dim);
    auto right_ibuffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightIndex_> > >(common_dim);
    auto right_ranges = tatami::create_container_of_Index_size<std::vector<tatami::SparseRange<RightValue_, RightIndex_> > >(common_dim);
    populate_sparse_buffers(true, common_dim, right_NC, right, right_vbuffers, right_ibuffers, right_ranges, options.num_threads);

    // If we're running in single-threaded mode and block size is not 1, we'll be writing directly to the output array.
    // In this case, the entire output array needs to be zeroed before iterating over the LHS matrix.
    const bool do_parallel = options.num_threads > 1;
    const bool needs_full_zero = !do_parallel && options.block_size != 1;

    // Figuring out which RHS columns have at least one structural non-zero.
    std::vector<RightIndex_> indices_to_transpose;
    {
        auto indices_to_transpose_bool = tatami::create_container_of_Index_size<std::vector<char> >(right_NC);
        for (LeftIndex_ cd = 0; cd < common_dim; ++cd) {
            const auto currange = right_ranges[cd];
            for (RightIndex_ x = 0; x < currange.number; ++x) {
                indices_to_transpose_bool[currange.index[x]] = 1;
            }
        }

        indices_to_transpose.reserve(right_NC);
        if (needs_full_zero) {
            for (LeftIndex_ rc = 0; rc < right_NC; ++rc) {
                if (indices_to_transpose_bool[rc]) {
                    indices_to_transpose.push_back(rc);
                }
            }
            std::fill_n(output, sanisizer::product_unsafe<std::size_t>(left_NR, right_NC), 0);

        } else {
            for (LeftIndex_ rc = 0; rc < right_NC; ++rc) {
                if (indices_to_transpose_bool[rc]) {
                    indices_to_transpose.push_back(rc);
                } else {
                    // Zeroing all of the output columns corresponding to RHS columns that have no non-zeros.
                    // We won't get another chance to do so because we'll never be touching these columns again.
                    std::fill_n(output + sanisizer::product_unsafe<std::size_t>(rc, left_NR), left_NR, 0);
                }
            }
        }
    }

    if (options.block_size == 1) {
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto ext = tatami::consecutive_extractor<false>(left, true, start, length);
            auto dbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(common_dim);

            // Use a temporary buffer to (i) improve data locality and (ii) avoid false sharing.
            auto tmp_row = tatami::create_container_of_Index_size<std::vector<Output_> >(right_NC);

            for (LeftIndex_ lr = 0; lr < length; ++lr) {
                const auto lptr = ext->fetch(dbuffer.data());
                for (LeftIndex_ cd = 0; cd < common_dim; ++cd) {
                    const auto rrange = right_ranges[cd];
                    const Output_ mult = lptr[cd];
                    for (RightIndex_ x = 0; x < rrange.number; ++x) {
                        tmp_row[rrange.index[x]] += mult * static_cast<Output_>(rrange.value[x]);
                    }
                }

                // We iterate over the positions of non-zeros, as these would have been modified by the innermost loop aboeve.
                // This ensures that we only transpose what we need to store.
                for (const auto rc : indices_to_transpose) {
                    output[sanisizer::nd_offset<std::size_t>(start + lr, left_NR, rc)] = tmp_row[rc];
                    tmp_row[rc] = 0;
                }
            }
        }, left_NR, options.num_threads);

    } else {
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto ext = tatami::consecutive_extractor<false>(left, true, start, length);

            const LeftIndex_ max_block_rows = sanisizer::min(length, options.block_size);
            std::vector<std::vector<LeftValue_> > lbuffers;
            lbuffers.reserve(max_block_rows);
            for (LeftIndex_ b = 0; b < max_block_rows; ++b) {
                lbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(common_dim));
            }
            auto lptrs = tatami::create_container_of_Index_size<std::vector<const LeftValue_*> >(max_block_rows);
            auto colbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(max_block_rows);

            // Create a temporary buffer to minimize false sharing during updates across all 'cd'.
            std::optional<std::vector<Output_> > tmp_cols;
            if (do_parallel) {
                tmp_cols.emplace(sanisizer::product<I<decltype(tmp_cols->size())> >(max_block_rows, right_NC));
            }

            LeftIndex_ lr = 0;
            while (lr < length) {
                const LeftIndex_ lr_num = sanisizer::min(options.block_size, length - lr);
                for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                    lptrs[lr_counter] = ext->fetch(lbuffers[lr_counter].data());
                }

                Output_* tmp_optr;
                LeftIndex_ out_row_offset;
                LeftIndex_ out_stride;
                if (do_parallel) {
                    tmp_optr = tmp_cols->data();
                    out_row_offset = 0;
                    out_stride = lr_num;
                } else {
                    tmp_optr = output;
                    out_row_offset = start + lr;
                    out_stride = left_NR;
                }

                for (LeftIndex_ cd = 0; cd < common_dim; ++cd) {
                    const auto rrange = right_ranges[cd];
                    if (rrange.number == 0) {
                        continue;
                    }

                    // Transfer this block of the 'cd'-th LHS column into a single buffer for a faster vector multiply-add in the inner loop.
                    // This rationale is unique to this function and is unlike any of the explanations in the sparse-blocking documentation.
                    for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                        colbuffer[lr_counter] = lptrs[lr_counter][cd];
                    }

                    for (RightIndex_ x = 0; x < rrange.number; ++x) {
                        const Output_ mult = rrange.value[x];
                        for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                            tmp_optr[sanisizer::nd_offset<std::size_t>(out_row_offset + lr_counter, out_stride, rrange.index[x])] += mult * static_cast<Output_>(colbuffer[lr_counter]);
                        }
                    }
                }

                if (do_parallel) {
                    // We only iterate over the non-zero positions, see explanation in the block_size == 1 case.
                    for (const auto rc : indices_to_transpose) {
                        const auto src = tmp_cols->data() + sanisizer::product_unsafe<std::size_t>(rc, lr_num);
                        std::copy_n(src, lr_num, output + sanisizer::nd_offset<std::size_t>(start + lr, left_NR, rc));
                        std::fill_n(src, lr_num, 0);
                    }
                }

                lr += lr_num;
            }
        }, left_NR, options.num_threads);
    }
}

}

#endif
