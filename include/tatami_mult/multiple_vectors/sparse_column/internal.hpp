#ifndef TATAMI_MULT_MULTIPLE_VECTORS_SPARSE_COLUMN_HPP
#define TATAMI_MULT_MULTIPLE_VECTORS_SPARSE_COLUMN_HPP

#include <cstddef>
#include <vector>
#include <optional>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../../utils.hpp"

namespace tatami_mult {

struct MultiplySparseColumnWithSomeVectorsOptions {
    int num_threads = 1;
    int block_size = 16;
};

template<typename Output_, typename Value_, typename Index_, typename Right_, typename GetOutput_>
void multiply_sparse_column_with_some_vectors_unblocked(
    const Index_ NR,
    const Index_ start,
    const Index_ length,
    const tatami::Matrix<Value_, Index_>& matrix,
    const std::vector<Right_*>& rhs_ptrs,
    GetOutput_ get_output
) {
    auto ext = tatami::consecutive_extractor<true>(&matrix, false, start, length);
    auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NR);
    auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NR);
    const auto num_rhs = rhs_ptrs.size();

    for (Index_ c = 0; c < length; ++c) {
        const auto range = ext->fetch(vbuffer.data(), ibuffer.data());
        for (I<decltype(num_rhs)> h = 0; h < num_rhs; ++h) {
            auto&& optr = get_output(h);
            const Output_ mult = rhs_ptrs[h][start + c];
            for (Index_ x = 0; x < range.number; ++x) {
                optr[range.index[x]] += mult * static_cast<Output_>(range.value[x]); 
            }
        }
    }
}

template<typename Output_, typename Value_, typename Index_, typename Right_, typename GetOutput_>
void multiply_sparse_column_with_some_vectors_blocked(
    const Index_ NR,
    const Index_ start,
    const Index_ length,
    const tatami::Matrix<Value_, Index_>& matrix,
    const std::vector<Right_*>& rhs_ptrs,
    GetOutput_ get_output,
    const Index_ block_size
) {
    // Our blocking strategy is to collect multiple LHS columns so that, for each RHS vector,
    // we can keep the corresponding output vector in cache for re-use with each LHS column.
    std::vector<std::vector<Value_> > vbuffers;
    std::vector<std::vector<Index_> > ibuffers;
    std::vector<tatami::SparseRange<Value_, Index_> > ranges;
    {
        const Index_ num_buffers = std::min(length, block_size);
        vbuffers.reserve(num_buffers);
        ibuffers.reserve(num_buffers);
        for (I<decltype(num_buffers)> b = 0; b < num_buffers; ++b) {
            vbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<Value_> >(NR));
            ibuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<Index_> >(NR));
        }
        sanisizer::resize(ranges, num_buffers);
    }

    const auto num_rhs = rhs_ptrs.size();
    auto ext = tatami::consecutive_extractor<true>(matrix, false, start, length);
    Index_ c = 0;
    while (c < length) {
        const Index_ cnum = std::min<Index_>(block_size, length - c);
        for (Index_ ccounter = 0; ccounter < cnum; ++ccounter) {
            ranges[ccounter] = ext->fetch(vbuffers[ccounter].data(), ibuffers[ccounter].data());
        }

        for (I<decltype(num_rhs)> h = 0; h < num_rhs; ++h) {
            auto&& outcol = get_output(h);
            for (auto ccounter = 0; ccounter < cnum; ++ccounter) {
                const auto& currange = ranges[ccounter];
                const Output_ mult = rhs_ptrs[h][start + c + ccounter];
                for (Index_ x = 0; x < currange.number; ++x) {
                    outcol[currange.index[x]] += mult * static_cast<Output_>(currange.value[x]);
                }
            }
        }
        c += cnum;
    }
}

template<typename Value_, typename Index_, typename Right_, typename GetOutput_>
void multiply_sparse_column_with_some_vectors(
    const tatami::Matrix<Value_, Index_>& matrix,
    const std::vector<Right_*>& rhs_ptrs,
    GetOutput_ get_output,
    const MultiplySparseColumnWithSomeVectorsOptions& options
) {
    const Index_ NR = matrix.nrow();
    const Index_ NC = matrix.ncol();
    const auto num_rhs = rhs_ptrs.size();

    typedef I<decltype(get_output(0)[0])> Output_;
    for (I<decltype(num_rhs)> c = 0; c < num_rhs; ++c) {
        std::fill_n(get_output(c), NR, 0);
    }

    const bool do_parallel = options.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<std::vector<Output_> > > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(options.num_threads - 1));
    }

    int num_used;
    if (options.block_size == 1) {
        num_used = tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
            if (!do_parallel || t == 0) {
                multiply_sparse_column_with_some_vectors_unblocked<Output_>(
                    NR,
                    start,
                    length,
                    matrix,
                    rhs_ptrs,
                    get_output
                );
                return;
            }

            auto tmp_out = sanisizer::create<std::vector<std::vector<Output_> > >(num_rhs);
            for (I<decltype(num_rhs)> c = 0; c < num_rhs; ++c) {
                tatami::resize_container_to_Index_size(tmp_out[c], NR);
            }

            multiply_sparse_column_with_some_vectors_unblocked<Output_>(
                NR,
                start,
                length,
                matrix,
                rhs_ptrs,
                [&](const I<decltype(num_rhs)> h) -> std::vector<Output_>& { return tmp_out[h]; }
            );

            if (do_parallel && t > 0) {
                (*tmp_results)[t - 1] = std::move(tmp_out);
            }
        }, NC, options.num_threads);

    } else {
        const auto block_size = sanisizer::cast<Index_>(options.block_size);
        num_used = tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
            if (!do_parallel || t == 0) {
                multiply_sparse_column_with_some_vectors_blocked<Output_>(
                    NR,
                    start,
                    length,
                    matrix,
                    rhs_ptrs,
                    get_output,
                    block_size
                );
                return;
            }

            auto tmp_out = sanisizer::create<std::vector<std::vector<Output_> > >(num_rhs);
            for (I<decltype(num_rhs)> c = 0; c < num_rhs; ++c) {
                tatami::resize_container_to_Index_size(tmp_out[c], NR);
            }

            multiply_sparse_column_with_some_vectors_blocked<Output_>(
                NR,
                start,
                length,
                matrix,
                rhs_ptrs,
                [&](const I<decltype(num_rhs)> h) -> std::vector<Output_>& { return tmp_out[h]; },
                block_size
            );

            if (do_parallel && t > 0) {
                (*tmp_results)[t - 1] = std::move(tmp_out);
            }
        }, NC, options.num_threads);
    }

    if (do_parallel) {
        for (int u = 1; u < num_used; ++u) {
            const auto& tmp = *((*tmp_results)[u - 1]);
            for (I<decltype(num_rhs)> j = 0; j < num_rhs; ++j) {
                const auto tmpvec = tmp[j];
                const auto outptr = get_output(j);
                for (Index_ r = 0; r < NR; ++r) {
                    outptr[r] += tmpvec[r];
                }
            }
        }
    }
}

}

#endif
