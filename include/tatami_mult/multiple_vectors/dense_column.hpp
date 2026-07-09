#ifndef TATAMI_MULT_MULTIPLE_VECTORS_DENSE_COLUMN_HPP
#define TATAMI_MULT_MULTIPLE_VECTORS_DENSE_COLUMN_HPP

#include <cstddef>
#include <vector>
#include <optional>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../utils.hpp"

/**
 * @file dense_column.hpp
 * @brief Dense column LHS, multiple vectors RHS.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_dense_column_with_multiple_vectors()`.
 */
struct MultiplyDenseColumnWithMultipleVectorsOptions {
    /**
     * Number of threads to use.
     */
    int num_threads = 1;

    /**
     * Primary block size.
     */
    int primary_block_size = 16;

    /**
     * Secondary block size.
     */
    int secondary_block_size = 64;
};

/**
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the matrix index.
 * @tparam Right_ Numeric type of the vector on the right hand side.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer column access, but will work with all matrices.
 * @param[in] right Vector of pointers, each of which points to an array of length `left.ncol()`.
 * Each entry contains a vector with which to multiply `left`.
 * @param[out] output Vector of pointers, each of which points to an array of length `left.nrow()`.
 * On output, the `i`-th entry stores the product `left * right[i]`.
 * @param options Further options.
 */
template<typename Value_, typename Index_, typename Right_, typename Output_>
void multiply_dense_column_with_multiple_vectors(
    const tatami::Matrix<Value_, Index_>& left,
    const std::vector<Right_*>& right,
    const std::vector<Output_*>& output,
    const MultiplyDenseColumnWithMultipleVectorsOptions& options
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

    const auto primary_block_size = sanisizer::cast<Index_>(options.primary_block_size);
    const auto secondary_block_size = sanisizer::cast<Index_>(options.secondary_block_size);

    const auto num_used = tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<false>(left, false, start, length);

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

        if (primary_block_size == 1) {
            auto buffer = tatami::create_container_of_Index_size<std::vector<Output_> >(NR);
            for (Index_ c = 0; c < length; ++c) {
                const auto ptr = ext->fetch(buffer.data());
                for (std::size_t j = 0; j < num_rhs; ++j) {
                    const auto optr = outptrs[j];
                    const Output_ mult = right[j][start + c];
                    for (Index_ r = 0; r < NR; ++r) {
                        optr[r] += mult * static_cast<Output_>(ptr[r]);
                    }
                }
            }

        } else {
            std::vector<std::vector<Value_> > buffers;
            std::vector<const Value_*> ptrs;
            {
                const Index_ max_block_cols = std::min(length, primary_block_size);
                buffers.reserve(max_block_cols);
                for (I<decltype(max_block_cols)> b = 0; b < max_block_cols; ++b) {
                    buffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<Value_> >(NR));
                }
                sanisizer::resize(ptrs, max_block_cols);
            }

            Index_ c = 0;
            while (c < length) {
                const Index_ cnum = std::min<Index_>(primary_block_size, length - c);
                for (Index_ ccounter = 0; ccounter < cnum; ++ccounter) {
                    ptrs[ccounter] = ext->fetch(buffers[ccounter].data());
                }

                std::size_t h = 0;
                while (h < num_rhs) {
                    // cast of primary_block_size to size_t is safe as primary_block_size must fit in an Index_ (and thus, by the tatami contract, a size_t).
                    const std::size_t hend = h + std::min<std::size_t>(primary_block_size, num_rhs - h);
                    Index_ r = 0;
                    while (r < NR) {
                        const Index_ rend = r + std::min<Index_>(secondary_block_size, NR - r);

                        for (auto ccounter = 0; ccounter < cnum; ++ccounter) {
                            const auto matcol = ptrs[ccounter];
                            for (auto hcopy = h; hcopy < hend; ++hcopy) {
                                const Output_ mult = right[hcopy][start + c + ccounter];
                                const auto outcol = outptrs[hcopy];
                                for (auto rcopy = r; rcopy < rend; ++rcopy) {
                                    outcol[rcopy] += mult * static_cast<Output_>(matcol[rcopy]);
                                }
                            }
                        }

                        r = rend;
                    }
                    h = hend;
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
