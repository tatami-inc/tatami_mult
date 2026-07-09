#ifndef TATAMI_MULT_MULTIPLE_VECTORS_SPARSE_COLUMN_HPP
#define TATAMI_MULT_MULTIPLE_VECTORS_SPARSE_COLUMN_HPP

#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"

#include "../utils.hpp"
#include "../sparse_dot_product.hpp"

/**
 * @file sparse_column.hpp
 * @brief Sparse column LHS, multiple vectors RHS.
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
    const Index_ NR = left.nrow();
    const Index_ NC = left.ncol();
    const auto num_rhs = right.size();
    typedef I<decltype(num_rhs)> RightIndex;
    for (RightIndex h = 0; h < num_rhs; ++h) {
        std::fill_n(output[h], NR, 0);
    }

    const bool do_parallel = options.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<std::vector<Output_> > > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(options.num_threads - 1));
    }

    const auto block_size = sanisizer::cast<Index_>(options.block_size);
    const auto num_used = tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(left, false, start, length);

        std::optional<std::vector<std::vector<Output_> > > tmp_output;
        std::optional<std::vector<Output_*> > tmp_outptrs;
        Output_* const* outptrs; 
        if (!do_parallel || t == 0) {
            outptrs = output.data();
        } else {
            tmp_output.emplace();
            tmp_output->reserve(sanisizer::cast<I<decltype(tmp_output->size())> >(num_rhs));
            for (RightIndex h = 0; h < num_rhs; ++h) {
                tmp_output->emplace_back(tatami::cast_Index_to_container_size<std::vector<Output_> >(NR));
            }
            tmp_outptrs.emplace();
            tmp_outptrs->reserve(sanisizer::cast<I<decltype(tmp_outptrs->size())> >(num_rhs));
            for (RightIndex h = 0; h < num_rhs; ++h) {
                tmp_outptrs->emplace_back((*tmp_output)[h].data());
            }
            outptrs = tmp_outptrs->data();
        }

        if (block_size == 1) {
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NR);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NR);
            for (Index_ c = 0; c < length; ++c) {
                const auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                for (RightIndex h = 0; h < num_rhs; ++h) {
                    const auto optr = outptrs[h];
                    const Output_ mult = right[h][start + c];
                    for (Index_ x = 0; x < range.number; ++x) {
                        optr[range.index[x]] += mult * static_cast<Output_>(range.value[x]); 
                    }
                }
            }

        } else {
            // Our blocking strategy is to collect multiple LHS columns so that, for each RHS vector,
            // we can keep the corresponding output vector in cache for re-use with each LHS column.
            std::vector<std::vector<Value_> > vbuffers;
            std::vector<std::vector<Index_> > ibuffers;
            std::vector<tatami::SparseRange<Value_, Index_> > ranges;
            {
                const Index_ max_block_rows = std::min(length, block_size);
                vbuffers.reserve(max_block_rows);
                ibuffers.reserve(max_block_rows);
                for (Index_ b = 0; b < max_block_rows; ++b) {
                    vbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<Value_> >(NR));
                    ibuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<Index_> >(NR));
                }
                sanisizer::resize(ranges, max_block_rows);
            }

            Index_ c = 0;
            while (c < length) {
                const Index_ cnum = std::min<Index_>(block_size, length - c);
                for (Index_ ccounter = 0; ccounter < cnum; ++ccounter) {
                    ranges[ccounter] = ext->fetch(vbuffers[ccounter].data(), ibuffers[ccounter].data());
                }
                for (RightIndex h = 0; h < num_rhs; ++h) {
                    const auto outcol = outptrs[h];
                    for (auto ccounter = 0; ccounter < cnum; ++ccounter) {
                        const auto& currange = ranges[ccounter];
                        const Output_ mult = right[h][start + c + ccounter];
                        for (Index_ x = 0; x < currange.number; ++x) {
                            outcol[currange.index[x]] += mult * static_cast<Output_>(currange.value[x]);
                        }
                    }
                }
                c += cnum;
            }
        }

        if (do_parallel && t > 0) {
            (*tmp_results)[t - 1] = std::move(tmp_output);
        }
    }, NC, options.num_threads);

    if (do_parallel) {
        for (int u = 1; u < num_used; ++u) {
            const auto& tmp = *((*tmp_results)[u - 1]);
            for (RightIndex h = 0; h < num_rhs; ++h) {
                const auto tmpvec = tmp[h];
                const auto outptr = output[h];
                for (Index_ r = 0; r < NR; ++r) {
                    outptr[r] += tmpvec[r];
                }
            }
        }
    }
}

}

#endif
