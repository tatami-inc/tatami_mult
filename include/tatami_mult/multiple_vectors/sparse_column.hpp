#ifndef TATAMI_MULT_MULTIPLE_VECTORS_SPARSE_COLUMN_HPP
#define TATAMI_MULT_MULTIPLE_VECTORS_SPARSE_COLUMN_HPP

#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"

#include "../utils.hpp"
#include "../sparse_dot_product.hpp"

/**
 * @file sparse_column.hpp
 * @brief Sparse column-major LHS, multiple vectors RHS.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_sparse_column_with_multiple_vectors()`.
 */
struct MultiplySparseColumnWithMultipleVectorsOptions {
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
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the matrix index.
 * @tparam Right_ Numeric type of the vector on the right hand side.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer column access, but will work with all matrices.
 * @param[in] right Vector of pointers, each of which points to an array of length `left.ncol()`.
 * Each entry contains a vector with which to multiply `left`.
 * @param[out] output Vector of pointers, each of which points to an array of length `left.nrow()`.
 * On output, the `i`-th entry stores the product `left * right[i]`.
 * @param options Further options.
 */
template<typename Value_, typename Index_, typename Right_, typename Output_>
void multiply_sparse_column_with_multiple_vectors(
    const tatami::Matrix<Value_, Index_>& left,
    const std::vector<Right_*>& right,
    const std::vector<Output_*>& output,
    const MultiplySparseColumnWithMultipleVectorsOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right.size();
    typedef I<decltype(right_NC)> RightIndex;
    for (RightIndex rc = 0; rc < right_NC; ++rc) {
        std::fill_n(output[rc], left_NR, 0);
    }

    const bool do_parallel = options.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<std::vector<Output_> > > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(options.num_threads - 1));
    }

    const auto num_used = tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(left, false, start, length);

        std::optional<std::vector<std::vector<Output_> > > tmp_output;
        std::optional<std::vector<Output_*> > tmp_outptrs;
        Output_* const* outptrs; 
        if (!do_parallel || t == 0) {
            outptrs = output.data();
        } else {
            tmp_output.emplace();
            tmp_output->reserve(sanisizer::cast<I<decltype(tmp_output->size())> >(right_NC));
            for (RightIndex rc = 0; rc < right_NC; ++rc) {
                tmp_output->emplace_back(tatami::cast_Index_to_container_size<std::vector<Output_> >(left_NR));
            }
            tmp_outptrs.emplace();
            tmp_outptrs->reserve(sanisizer::cast<I<decltype(tmp_outptrs->size())> >(right_NC));
            for (RightIndex rc = 0; rc < right_NC; ++rc) {
                tmp_outptrs->emplace_back((*tmp_output)[rc].data());
            }
            outptrs = tmp_outptrs->data();
        }

        if (options.block_size == 1) {
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(left_NR);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(left_NR);
            for (Index_ cd = 0; cd < length; ++cd) {
                const auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                for (RightIndex rc = 0; rc < right_NC; ++rc) {
                    const auto optr = outptrs[rc];
                    const Output_ mult = right[rc][start + cd];
                    for (Index_ x = 0; x < range.number; ++x) {
                        optr[range.index[x]] += mult * static_cast<Output_>(range.value[x]); 
                    }
                }
            }

        } else {
            // Our blocking strategy is to collect multiple LHS columns so that, for each RHS vector,
            // we can keep the corresponding output vector in cache for re-use with each LHS column.
            std::vector<std::vector<Value_> > left_vbuffers;
            std::vector<std::vector<Index_> > left_ibuffers;
            std::vector<tatami::SparseRange<Value_, Index_> > left_ranges;
            {
                const Index_ max_block_cols = sanisizer::min(length, options.block_size);
                left_vbuffers.reserve(max_block_cols);
                left_ibuffers.reserve(max_block_cols);
                for (Index_ cd = 0; cd < max_block_cols; ++cd) {
                    left_vbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<Value_> >(left_NR));
                    left_ibuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<Index_> >(left_NR));
                }
                sanisizer::resize(left_ranges, max_block_cols);
            }

            Index_ cd = 0;
            while (cd < length) {
                const Index_ cd_num = sanisizer::min(options.block_size, length - cd);
                for (Index_ cd_counter = 0; cd_counter < cd_num; ++cd_counter) {
                    left_ranges[cd_counter] = ext->fetch(left_vbuffers[cd_counter].data(), left_ibuffers[cd_counter].data());
                }
                for (RightIndex rc = 0; rc < right_NC; ++rc) {
                    const auto outcol = outptrs[rc];
                    for (Index_ cd_counter = 0; cd_counter < cd_num; ++cd_counter) {
                        const auto& currange = left_ranges[cd_counter];
                        const Output_ mult = right[rc][start + cd + cd_counter];
                        for (Index_ x = 0; x < currange.number; ++x) {
                            outcol[currange.index[x]] += mult * static_cast<Output_>(currange.value[x]);
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
            for (RightIndex rc = 0; rc < right_NC; ++rc) {
                const auto& tmpvec = tmp[rc];
                const auto outptr = output[rc];
                for (Index_ lr = 0; lr < left_NR; ++lr) {
                    outptr[lr] += tmpvec[lr];
                }
            }
        }
    }
}

}

#endif
