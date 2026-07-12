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
 * @brief Dense column-major LHS, multiple vectors RHS.
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
        auto ext = tatami::consecutive_extractor<false>(left, false, start, length);

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

        if (options.primary_block_size == 1) {
            auto buffer = tatami::create_container_of_Index_size<std::vector<Output_> >(left_NR);
            for (Index_ cd = 0; cd < length; ++cd) {
                const auto ptr = ext->fetch(buffer.data());
                for (RightIndex rc = 0; rc < right_NC; ++rc) {
                    const auto optr = outptrs[rc];
                    const Output_ mult = right[rc][start + cd];
                    for (Index_ lr = 0; lr < left_NR; ++lr) {
                        optr[lr] += mult * static_cast<Output_>(ptr[lr]);
                    }
                }
            }

        } else {
            std::vector<std::vector<Value_> > left_buffers;
            std::vector<const Value_*> left_ptrs;
            {
                const Index_ max_block_cols = sanisizer::min(length, options.primary_block_size);
                left_buffers.reserve(max_block_cols);
                for (Index_ cd = 0; cd < max_block_cols; ++cd) {
                    left_buffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<Value_> >(left_NR));
                }
                sanisizer::resize(left_ptrs, max_block_cols);
            }

            Index_ cd = 0;
            while (cd < length) {
                const Index_ cd_num = sanisizer::min(options.primary_block_size, length - cd);
                for (Index_ cd_counter = 0; cd_counter < cd_num; ++cd_counter) {
                    left_ptrs[cd_counter] = ext->fetch(left_buffers[cd_counter].data());
                }

                RightIndex rc = 0;
                while (rc < right_NC) {
                    const RightIndex rc_end = rc + sanisizer::min(options.primary_block_size, right_NC - rc);
                    Index_ lr = 0;
                    while (lr < left_NR) {
                        const Index_ lr_end = lr + sanisizer::min(options.secondary_block_size, left_NR - lr);

                        for (Index_ cd_counter = 0; cd_counter < cd_num; ++cd_counter) {
                            const auto matcol = left_ptrs[cd_counter];
                            for (auto rc_copy = rc; rc_copy < rc_end; ++rc_copy) {
                                const Output_ mult = right[rc_copy][start + cd + cd_counter];
                                const auto outcol = outptrs[rc_copy];
                                for (auto lr_copy = lr; lr_copy < lr_end; ++lr_copy) {
                                    outcol[lr_copy] += mult * static_cast<Output_>(matcol[lr_copy]);
                                }
                            }
                        }

                        lr = lr_end;
                    }
                    rc = rc_end;
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
