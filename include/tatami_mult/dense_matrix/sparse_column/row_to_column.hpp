#ifndef TATAMI_MULT_DENSE_MATRIX_SPARSE_ROW_ROW_TO_COLUMN_HPP
#define TATAMI_MULT_DENSE_MATRIX_SPARSE_ROW_ROW_TO_COLUMN_HPP

#include <cstddef>
#include <vector>
#include <optional>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../../utils.hpp"

/**
 * @file column_to_column.hpp
 * @brief Sparse column LHS, dense column-major matrix RHS, column-major output.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_sparse_column_with_dense_row_matrix_to_column_output()`.
 */
struct MultiplySparseColumnWithDenseRowMatrixToColumnOutputOptions {
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
 * On output, this contains the product `left * right` in column-major order.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_sparse_column_with_dense_row_matrix_to_column_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplySparseColumnWithDenseRowMatrixToColumnOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right.ncol();

    const bool do_parallel = options.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<Output_> > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(options.num_threads - 1));
    }

    auto right_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(right_NC);
    auto right_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(right_NC);
    populate_dense_buffers(false, right_NC, common_dim, right, right_buffers, right_ptrs, options.num_threads);

    const auto num_used = tatami::parallelize([&](int t, LeftIndex_ start, LeftIndex_ length) -> void {
        auto left_ext = tatami::consecutive_extractor<true>(left, false, start, length);
        auto right_ext = tatami::consecutive_extractor<false>(right, true, start, length);

        std::optional<std::vector<Output_> > tmp_output;
        Output_* outptr; 
        if (!do_parallel || t == 0) {
            outptr = output;
        } else {
            tmp_output.emplace(sanisizer::product<I<decltype(tmp_output->size())> >(left_NR, right_NC));
            outptr = tmp_output->data();
        }

        if (options.block_size == 1) {
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(left_NR);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<LeftIndex_> >(left_NR);
            auto dbuffer = tatami::create_container_of_Index_size<std::vector<RightValue_> >(right_NC);

            for (LeftIndex_ cd = 0; cd < length; ++cd) {
                const auto lrange = left_ext->fetch(vbuffer.data(), ibuffer.data());
                const auto rptr = right_ext->fetch(dbuffer.data());
                for (RightIndex_ rc = 0; rc < right_NC; ++rc) {
                    const Output_ mult = rptr[rc];
                    for (LeftIndex_ x = 0; x < lrange.number; ++x) {
                        outptr[sanisizer::nd_offset<std::size_t>(lrange.index[x], left_NR, rc)] += mult * static_cast<Output_>(lrange.value[x]); 
                    }
                }
            }

        } else {
            // Our blocking strategy is to collect multiple LHS columns so that, for each RHS vector,
            // we can keep the corresponding output vector in cache for re-use with each LHS column.
            std::vector<std::vector<LeftValue_> > left_vbuffers;
            std::vector<std::vector<LeftIndex_> > left_ibuffers;
            std::vector<tatami::SparseRange<LeftValue_, LeftIndex_> > left_ranges;
            std::vector<std::vector<RightValue_> > right_dbuffers;
            std::vector<const RightValue_*> right_ptrs;
            {
                const LeftIndex_ max_block_cols = sanisizer::min(length, options.block_size);
                left_vbuffers.reserve(max_block_cols);
                left_ibuffers.reserve(max_block_cols);
                right_dbuffers.reserve(max_block_cols);
                for (LeftIndex_ cd = 0; cd < max_block_cols; ++cd) {
                    left_vbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(left_NR));
                    left_ibuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftIndex_> >(left_NR));
                    right_dbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<RightValue_> >(right_NC));
                }
                sanisizer::resize(left_ranges, max_block_cols);
                sanisizer::resize(right_ptrs, max_block_cols);
            }

            LeftIndex_ cd = 0;
            while (cd < length) {
                const LeftIndex_ cd_num = sanisizer::min(options.block_size, length - cd);
                for (LeftIndex_ cd_counter = 0; cd_counter < cd_num; ++cd_counter) {
                    left_ranges[cd_counter] = left_ext->fetch(left_vbuffers[cd_counter].data(), left_ibuffers[cd_counter].data());
                    right_ptrs[cd_counter] = right_ext->fetch(right_dbuffers[cd_counter].data());
                }
                for (RightIndex_ rc = 0; rc < right_NC; ++rc) {
                    for (LeftIndex_ cd_counter = 0; cd_counter < cd_num; ++cd_counter) {
                        const auto& currange = left_ranges[cd_counter];
                        const Output_ mult = right_ptrs[cd_counter][rc];
                        for (LeftIndex_ x = 0; x < currange.number; ++x) {
                            outptr[sanisizer::nd_offset<std::size_t>(currange.index[x], left_NR, rc)] += mult * static_cast<Output_>(currange.value[x]);
                        }
                    }
                }
                cd += cd_num;
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
