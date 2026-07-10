#ifndef TATAMI_MULT_DENSE_MATRIX_DENSE_ROW_COLUMN_TO_ROW_HPP
#define TATAMI_MULT_DENSE_MATRIX_DENSE_ROW_COLUMN_TO_ROW_HPP

#include <vector>
#include <cstddef>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../../dense_dot_product.hpp"
#include "../../utils.hpp"

namespace tatami_mult {

struct MultiplyDenseRowWithDenseColumnMatrixToRowOutputOptions {
    int num_threads = 1;
    int primary_block_size = 16;
    int secondary_block_size = 64;
};

template<std::size_t accumulators_ = 4, typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_dense_row_with_dense_column_matrix_to_row_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const std::vector<const RightValue_*>& right,
    const RightIndex_ num_rhs,
    Output_* const output,
    const MultiplyDenseRowWithDenseColumnMatrixToRowOutputOptions& options
) {
    const auto NR = left.nrow();
    const auto NC = left.ncol();

    if (options.primary_block_size == 1) {
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto lext = tatami::consecutive_extractor<false>(left, true, start, length);
            auto lbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(NC);
            for (LeftIndex_ r = 0; r < length; ++r) {
                const auto lptr = lext->fetch(lbuffer.data());
                for (RightIndex_ h = 0; h < num_rhs; ++h) {
                    output[sanisizer::nd_offset<std::size_t>(h, num_rhs, start + r)] = dense_dot_product<accumulators_>(NC, lptr, right[h], static_cast<Output_>(0));
                }
            }
        }, NR, options.num_threads);
        return;
    } 

    const auto primary_block_size = sanisizer::cast<LeftIndex_>(options.primary_block_size);
    const auto secondary_block_size = sanisizer::cast<LeftIndex_>(options.secondary_block_size);

    const bool do_parallel = options.num_threads > 1;
    if (!do_parallel) {
        std::fill_n(output, sanisizer::product_unsafe<std::size_t>(NR, num_rhs), 0);
    }

    tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
        auto ext = tatami::consecutive_extractor<false>(left, true, start, length);

        const LeftIndex_ max_block_rows = std::min(length, primary_block_size);
        std::vector<std::vector<LeftValue_> > lbuffers;
        lbuffers.reserve(max_block_rows);
        for (LeftIndex_ b = 0; b < max_block_rows; ++b) {
            lbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(NC));
        }
        auto lptrs = tatami::create_container_of_Index_size<std::vector<const LeftValue_*> >(max_block_rows);

        std::optional<std::vector<Output_> > tmp_output;
        Output_* optr;
        if (do_parallel) {
            // For the multi-threaded case, we create some temporary buffers to hold the partial dot products for the current set of submatrices.
            // This aims to mitigate false sharing as we update each block's partial dot products in the loop over the common dimension.
            // There is still some potential for false sharing when we transfer the results to the output buffers,
            // but this is the same as the unblocked case so we won't worry about it.
            const auto max_block_cols = sanisizer::min(num_rhs, primary_block_size);
            const auto max_block_rows = std::min(length, primary_block_size);
            tmp_output.emplace(sanisizer::product<I<decltype(tmp_output->size())> >(max_block_cols, max_block_rows));
            optr = tmp_output->data();
        } else {
            optr = output; // no need to add 'start' as this is zero in the single-threaded case.
        }

        LeftIndex_ r = 0;
        while (r < length) {
            const LeftIndex_ rnum = std::min<LeftIndex_>(primary_block_size, length - r);
            for (LeftIndex_ rcounter = 0; rcounter < rnum; ++rcounter) {
                lptrs[rcounter] = ext->fetch(lbuffers[rcounter].data());
            }

            RightIndex_ h = 0;
            while (h < num_rhs) {
                const RightIndex_ hnum = sanisizer::min(primary_block_size, num_rhs - h);

                RightIndex_ row_offset, col_offset, stride;
                if (do_parallel) {
                    std::fill_n(optr, sanisizer::product_unsafe<std::size_t>(hnum, rnum), 0);
                    row_offset = 0;
                    col_offset = 0;
                    stride = hnum;
                } else {
                    row_offset = r; // start == 0 in serial code, so no need to add start here.
                    col_offset = h;
                    stride = num_rhs;
                }

                LeftIndex_ c = 0;
                while (c < NC) {
                    const LeftIndex_ cnum = std::min<LeftIndex_>(secondary_block_size, NC - c);
                    for (LeftIndex_ rcounter = 0; rcounter < rnum; ++rcounter) {
                        const auto leftrow = lptrs[rcounter];
                        for (RightIndex_ hcounter = 0; hcounter < hnum; ++hcounter) {
                            auto& dest = optr[sanisizer::nd_offset<std::size_t>(col_offset + hcounter, stride, row_offset + rcounter)]; 
                            dest = dense_dot_product<accumulators_>(
                                cnum,
                                right[h + hcounter] + c,
                                leftrow + c,
                                dest
                            );
                        }
                    }
                    c += cnum;
                }

                if (do_parallel) {
                    for (LeftIndex_ rcounter = 0; rcounter < rnum; ++rcounter) {
                        std::copy_n(
                            optr + sanisizer::product_unsafe<std::size_t>(hnum, rcounter),
                            hnum,
                            output + sanisizer::nd_offset<std::size_t>(h, num_rhs, start + r + rcounter)
                        );
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
