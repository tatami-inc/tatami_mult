#ifndef TATAMI_MULT_MULTIPLE_VECTORS_SPARSE_ROW_INTERNAL_HPP
#define TATAMI_MULT_MULTIPLE_VECTORS_SPARSE_ROW_INTERNAL_HPP

#include <vector>
#include <cstddef>

#include "tatami/tatami.hpp"

#include "../../utils.hpp"
#include "../../sparse_dot_product.hpp"

namespace tatami_mult {

struct MultiplySparseRowWithSomeVectorsOptions {
    int num_threads = 1;
    int block_size = 16;
};

template<std::size_t accumulators_, typename Value_, typename Index_, typename Right_, typename GetOutput_>
void multiply_sparse_row_with_some_vectors(
    const tatami::Matrix<Value_, Index_>& matrix,
    const std::vector<Right_*>& rhs_ptrs,
    GetOutput_ get_output,
    const MultiplySparseRowWithSomeVectorsOptions& options
) {
    const Index_ NR = matrix.nrow();
    const Index_ NC = matrix.ncol();
    const auto num_rhs = rhs_ptrs.size();
    typedef I<decltype(get_output(0)[0])> Output;

    if (options.block_size == 1) {
        tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
            auto ext = tatami::consecutive_extractor<true>(&matrix, true, start, length);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);

            for (Index_ r = 0; r < length; ++r) {
                const auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                for (I<decltype(num_rhs)> h = 0; h < num_rhs; ++h) {
                    const auto rptr = rhs_ptrs[h];
                    auto&& optr = get_output(h);
                    // Some false sharing potential here, but we just touch each location once per outer loop, so it's fine.
                    optr[start + r] = dense_sparse_dot_product<Output>(rptr, range);
                }
            }
        }, NR, options.num_threads);

    } else {
        const Index_ block_size = sanisizer::cast<Index_>(options.block_size);
        tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
            auto ext = tatami::consecutive_extractor<true>(&matrix, true, start, length);

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
                    const auto rcol = rhs_ptrs[h];
                    auto&& outcol = get_output(h);
                    for (Index_ rcounter = 0; rcounter < rnum; ++rcounter) {
                        const auto& currange = ranges[rcounter];
                        outcol[start + r + rcounter] = sparse_dot_product<accumulators_>(
                            currange.number,
                            currange.value,
                            currange.index,
                            rcol,
                            static_cast<Output>(0)
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
