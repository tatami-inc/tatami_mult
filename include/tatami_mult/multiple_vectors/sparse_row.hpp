#ifndef TATAMI_MULT_MULTIPLE_VECTORS_SPARSE_ROW_HPP
#define TATAMI_MULT_MULTIPLE_VECTORS_SPARSE_ROW_HPP

#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"

#include "../utils.hpp"
#include "../sparse_dot_product.hpp"

/**
 * @file sparse_row.hpp
 * @brief Sparse row-major LHS, multiple vectors RHS.
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
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right.size();
    typedef I<decltype(right_NC)> RightIndex;

    if (options.block_size == 1) {
        tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
            auto ext = tatami::consecutive_extractor<true>(left, true, start, length);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(common_dim);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(common_dim);

            for (Index_ lr = 0; lr < length; ++lr) {
                const auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                for (RightIndex rc = 0; rc < right_NC; ++rc) {
                    // Implicit cast of range.number to size_t is safe, as per the tatami contract.
                    // Also some false sharing potential here, but we just touch each location once per outer loop, so it's fine.
                    output[rc][start + lr] = sparse_dot_product<accumulators_>(range.number, range.value, range.index, right[rc], static_cast<Output_>(0));
                }
            }
        }, left_NR, options.num_threads);

    } else {
        tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
            auto ext = tatami::consecutive_extractor<true>(left, true, start, length);

            // Our blocking strategy is to collect multiple LHS rows so that, for each RHS vector,
            // we can keep it in cache for easy re-use when computing the dot-product for each LHS row.
            std::vector<std::vector<Value_> > left_vbuffers;
            std::vector<std::vector<Index_> > left_ibuffers;
            std::vector<tatami::SparseRange<Value_, Index_> > left_ranges;
            {
                const Index_ max_block_rows = sanisizer::min(length, options.block_size);
                left_vbuffers.reserve(max_block_rows);
                left_ibuffers.reserve(max_block_rows);
                for (Index_ lr = 0; lr < max_block_rows; ++lr) {
                    left_vbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<Value_> >(common_dim));
                    left_ibuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<Index_> >(common_dim));
                }
                sanisizer::resize(left_ranges, max_block_rows);
            }

            Index_ lr = 0;
            while (lr < length) {
                const Index_ lr_num = sanisizer::min(options.block_size, length - lr);
                for (Index_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                    left_ranges[lr_counter] = ext->fetch(left_vbuffers[lr_counter].data(), left_ibuffers[lr_counter].data());
                }

                for (RightIndex rc = 0; rc < right_NC; ++rc) {
                    const auto rcol = right[rc];
                    const auto outcol = output[rc];
                    for (Index_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                        const auto& currange = left_ranges[lr_counter];
                        outcol[start + lr + lr_counter] = sparse_dot_product<accumulators_>(
                            currange.number, // Implicit cast of range.number to size_t is safe, as per the tatami contract.
                            currange.value,
                            currange.index,
                            rcol,
                            static_cast<Output_>(0)
                        );
                    }
                }

                lr += lr_num;
            }
        }, left_NR, options.num_threads);
    }
}

}

#endif
