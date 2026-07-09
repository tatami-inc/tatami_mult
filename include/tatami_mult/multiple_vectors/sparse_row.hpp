#ifndef TATAMI_MULT_MULTIPLE_VECTORS_SPARSE_ROW_HPP
#define TATAMI_MULT_MULTIPLE_VECTORS_SPARSE_ROW_HPP

#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"

#include "../utils.hpp"
#include "../sparse_dot_product.hpp"

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
    const Index_ NR = left.nrow();
    const Index_ NC = left.ncol();
    const auto num_rhs = right.size();

    if (options.block_size == 1) {
        tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
            auto ext = tatami::consecutive_extractor<true>(left, true, start, length);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);

            for (Index_ r = 0; r < length; ++r) {
                const auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                for (I<decltype(num_rhs)> h = 0; h < num_rhs; ++h) {
                    // Some false sharing potential here, but we just touch each location once per outer loop, so it's fine.
                    output[h][start + r] = sparse_dot_product<accumulators_>(range.number, range.value, range.index, right[h], static_cast<Output_>(0));
                }
            }
        }, NR, options.num_threads);

    } else {
        const Index_ block_size = sanisizer::cast<Index_>(options.block_size);
        tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
            auto ext = tatami::consecutive_extractor<true>(left, true, start, length);

            // Our blocking strategy is to collect multiple LHS rows so that, for each RHS vector,
            // we can keep it in cache for easy re-use when computing the dot-product for each LHS row.
            std::vector<std::vector<Value_> > vbuffers;
            std::vector<std::vector<Index_> > ibuffers;
            std::vector<tatami::SparseRange<Value_, Index_> > ranges;
            {
                const Index_ num_buffers = std::min(length, block_size);
                vbuffers.reserve(num_buffers);
                ibuffers.reserve(num_buffers);
                for (I<decltype(num_buffers)> b = 0; b < num_buffers; ++b) {
                    vbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<Value_> >(NC));
                    ibuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<Index_> >(NC));
                }
                sanisizer::resize(ranges, num_buffers);
            }

            Index_ r = 0;
            while (r < length) {
                const Index_ rnum = std::min<Index_>(block_size, length - r);
                for (Index_ rcounter = 0; rcounter < rnum; ++rcounter) {
                    ranges[rcounter] = ext->fetch(vbuffers[rcounter].data(), ibuffers[rcounter].data());
                }

                for (I<decltype(num_rhs)> h = 0; h < num_rhs; ++h) {
                    const auto rcol = right[h];
                    const auto outcol = output[h];
                    for (Index_ rcounter = 0; rcounter < rnum; ++rcounter) {
                        const auto& currange = ranges[rcounter];
                        outcol[start + r + rcounter] = sparse_dot_product<accumulators_>(
                            currange.number,
                            currange.value,
                            currange.index,
                            rcol,
                            static_cast<Output_>(0)
                        );
                    }
                }

                r += rnum;
            }
        }, NR, options.num_threads);
    }
}

}

#endif
