#ifndef TATAMI_MULT_MULTIPLE_VECTORS_DENSE_ROW_INTERNAL_HPP
#define TATAMI_MULT_MULTIPLE_VECTORS_DENSE_ROW_INTERNAL_HPP

#include <vector>
#include <cstddef>

#include "tatami/tatami.hpp"

#include "../../utils.hpp"
#include "../../dense_dot_product.hpp"

namespace tatami_mult {

struct MultiplyDenseRowWithSomeVectorsOptions {
    int num_threads = 1;
    int primary_block_size = 16;
    int secondary_block_size = 64;
};

template<std::size_t accumulators_, typename Value_, typename Index_, typename Right_, typename GetOutput_>
void multiply_dense_row_with_some_vectors(
    const tatami::Matrix<Value_, Index_>& matrix,
    const std::vector<Right_*>& rhs_ptrs,
    GetOutput_ get_output,
    const MultiplyDenseRowWithSomeVectorsOptions& options
) {
    const auto num_rhs = rhs_ptrs.size();
    const Index_ NR = matrix.nrow();
    const Index_ NC = matrix.ncol();
    typedef I<decltype(get_output(0)[0])> Output;

    if (options.primary_block_size == 1) {
        tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
            auto ext = tatami::consecutive_extractor<false>(&matrix, true, start, length);
            auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
            for (Index_ r = 0; r < length; ++r) {
                const auto ptr = ext->fetch(buffer.data());
                for (I<decltype(num_rhs)> j = 0; j < num_rhs; ++j) {
                    const auto rptr = rhs_ptrs[j];
                    auto&& optr = get_output(j);
                    optr[start + r] = std::inner_product(ptr, ptr + NC, rptr, static_cast<Output>(0));
                }
            }
        }, NR, options.num_threads);

    } else {
        const auto primary_block_size = sanisizer::cast<Index_>(options.primary_block_size);
        const auto secondary_block_size = sanisizer::cast<Index_>(options.secondary_block_size);

        tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
            auto ext = tatami::consecutive_extractor<false>(&matrix, true, start, length);

            std::vector<std::vector<Value_> > buffers;
            std::vector<const Value_*> ptrs;
            {
                const Index_ num_buffers = std::min(length, primary_block_size);
                buffers.reserve(num_buffers);
                for (I<decltype(num_buffers)> b = 0; b < num_buffers; ++b) {
                    buffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<Value_> >(NC));
                }
                sanisizer::resize(ptrs, num_buffers);
            }

            Index_ r = 0;
            while (r < length) {
                const Index_ rnum = std::min<Index_>(primary_block_size, length - r);
                for (Index_ rcounter = 0; rcounter < rnum; ++rcounter) {
                    ptrs[rcounter] = ext->fetch(buffers[rcounter].data());
                }

                I<decltype(num_rhs)> h = 0;
                while (h < num_rhs) {
                    // cast of primary_block_size to size_t is safe as primary_block_size must fit in an Index_ (and thus, by the tatami contract, a size_t).
                    const I<decltype(num_rhs)> hend = h + std::min<std::size_t>(primary_block_size, num_rhs - h);
                    Index_ c = 0;
                    while (c < NC) {
                        const Index_ cnum = std::min<Index_>(secondary_block_size, NC - c);

                        for (auto hcopy = h; hcopy < hend; ++hcopy) {
                            auto&& outcol = get_output(hcopy);
                            const auto& rightcol = rhs_ptrs[hcopy];
                            for (auto rcounter = 0; rcounter < rnum; ++rcounter) {
                                auto& dest = outcol[start + r + rcounter];
                                dest = dense_dot_product<accumulators_>(
                                    cnum,
                                    rightcol + c,
                                    ptrs[rcounter] + c,
                                    dest
                                );
                            }
                        }

                        c += cnum;
                    }
                    h = hend;
                }
                r += rnum;
            }
        }, NR, options.num_threads);
    }
}

}

#endif
