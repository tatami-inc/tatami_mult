#ifndef TATAMI_MULT_MULTIPLE_VECTORS_DENSE_ROW_HPP
#define TATAMI_MULT_MULTIPLE_VECTORS_DENSE_ROW_HPP

#include <cstddef>
#include <vector>
#include <optional>

#include "tatami/tatami.hpp"

#include "../utils.hpp"
#include "../dense_dot_product.hpp"

/**
 * @file dense_row.hpp
 * @brief Dense row LHS, multiple vectors RHS.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_dense_row_with_multiple_vectors()`.
 */
struct MultiplyDenseRowWithMultipleVectorsOptions {
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
 * @tparam accumulators_ Number of accumulators for computing the dot product.
 * This should be positive and is very often a power of 2, with values of 2-8 typically providing some performance improvement on modern CPUs.
 * Different numbers of accumulators may result in slight changes to the output due to changes in floating-point round-off error.
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the matrix index.
 * @tparam Right_ Numeric type of the vector on the right hand side.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * @param[in] right Vector of pointers, each of which points to an array of length `left.ncol()`.
 * Each entry contains a vector with which to multiply `left`.
 * @param[out] output Vector of pointers, each of which points to an array of length `left.nrow()`.
 * On output, the `i`-th entry stores the product `left * right[i]`.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename Value_, typename Index_, typename Right_, typename Output_>
void multiply_dense_row_with_multiple_vectors(
    const tatami::Matrix<Value_, Index_>& left,
    const std::vector<Right_*>& right,
    const std::vector<Output_*>& output,
    const MultiplyDenseRowWithMultipleVectorsOptions& options
) {
    const auto NR = left.nrow();
    const auto NC = left.ncol();
    const auto num_rhs = right.size();
    typedef I<decltype(num_rhs)> RightIndex;

    if (options.primary_block_size == 1) {
        tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
            auto lext = tatami::consecutive_extractor<false>(left, true, start, length);
            auto lbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
            for (Index_ r = 0; r < length; ++r) {
                const auto lptr = lext->fetch(lbuffer.data());
                for (RightIndex h = 0; h < num_rhs; ++h) {
                    output[h][start + r] = dense_dot_product<accumulators_>(NC, lptr, right[h], static_cast<Output_>(0));
                }
            }
        }, NR, options.num_threads);
        return;
    } 

    const auto primary_block_size = sanisizer::cast<Index_>(options.primary_block_size);
    const auto secondary_block_size = sanisizer::cast<Index_>(options.secondary_block_size);

    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<false>(left, true, start, length);

        const Index_ max_block_rows = std::min(length, primary_block_size);
        std::vector<std::vector<Value_> > lbuffers;
        lbuffers.reserve(max_block_rows);
        for (Index_ b = 0; b < max_block_rows; ++b) {
            lbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<Value_> >(NC));
        }
        auto lptrs = tatami::create_container_of_Index_size<std::vector<const Value_*> >(max_block_rows);

        // Creating a block of buffers to hold the partial dot products for the current set of submatrices as we iterate over the columns.
        // This aims to mitigate false sharing as we update each block's partial dot products.
        // There is still some potential for false sharing when we transfer the results to the output buffers,
        // but this is the same as the unblocked case so we won't worry about it.
        std::optional<std::vector<std::vector<Output_> > > tmp_output;
        std::optional<std::vector<Output_*> > tmp_outptrs;
        if (start > 0) {
            const RightIndex max_block_cols = sanisizer::min(num_rhs, primary_block_size);
            const RightIndex max_block_rows = sanisizer::min(length, primary_block_size);
            tmp_output.emplace();
            tmp_output->reserve(max_block_cols);
            for (RightIndex h = 0; h < max_block_cols; ++h) {
                tmp_output->emplace_back(tatami::cast_Index_to_container_size<std::vector<Output_> >(max_block_rows));
            }
            tmp_outptrs.emplace();
            tmp_outptrs->reserve(max_block_cols);
            for (RightIndex h = 0; h < max_block_cols; ++h) {
                tmp_outptrs->emplace_back((*tmp_output)[h].data());
            }
        }

        Index_ r = 0;
        while (r < length) {
            const Index_ rnum = std::min<Index_>(primary_block_size, length - r);
            for (Index_ rcounter = 0; rcounter < rnum; ++rcounter) {
                lptrs[rcounter] = ext->fetch(lbuffers[rcounter].data());
            }

            RightIndex h = 0;
            while (h < num_rhs) {
                // cast of primary_block_size to size_t is safe as primary_block_size must fit in an Index_ (and thus, by the tatami contract, a size_t).
                const auto hnum = std::min<std::size_t>(primary_block_size, num_rhs - h);

                Output_* const * outptrs;
                Index_ row_offset;
                if (start > 0) {
                    outptrs = tmp_outptrs->data();
                    row_offset = 0; // no need to add start, as it's ignored when saving to tmp_output.
                } else {
                    outptrs = output.data() + h;
                    row_offset = r; // no need to add start, as it's zero. 
                }
                // Zeroing all of the buffers prior to accumulation.
                for (std::size_t hcounter = 0; hcounter < hnum; ++hcounter) {
                    std::fill_n(outptrs[hcounter] + row_offset, rnum, 0);
                }

                Index_ c = 0;
                while (c < NC) {
                    const Index_ cnum = std::min<Index_>(secondary_block_size, NC - c);
                    for (std::size_t hcounter = 0; hcounter < hnum; ++hcounter) {
                        const auto outcol = outptrs[hcounter];
                        const auto& rightcol = right[h + hcounter];
                        for (Index_ rcounter = 0; rcounter < rnum; ++rcounter) {
                            auto& dest = outcol[row_offset + rcounter]; 
                            dest = dense_dot_product<accumulators_>(
                                cnum,
                                rightcol + c,
                                lptrs[rcounter] + c,
                                dest
                            );
                        }
                    }
                    c += cnum;
                }

                if (start > 0) {
                    for (std::size_t hcounter = 0; hcounter < hnum; ++hcounter) {
                        const auto& src = (*tmp_output)[hcounter];
                        std::copy_n(src.begin(), rnum, output[h + hcounter] + start + r);
                    }
                }

                h += hnum;
            }
            r += rnum;
        }
    }, NR, options.num_threads);
}

}

#endif
