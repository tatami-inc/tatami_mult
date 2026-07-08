#ifndef TATAMI_MULT_MULTIPLE_VECTORS_DENSE_COLUMN_HPP
#define TATAMI_MULT_MULTIPLE_VECTORS_DENSE_COLUMN_HPP

#include <cstddef>
#include <vector>
#include <optional>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../../utils.hpp"

namespace tatami_mult {

struct MultiplyDenseColumnWithSomeVectorsOptions {
    int num_threads = 1;
    int primary_block_size = 16;
    int secondary_block_size = 64;
};

template<typename Output_, typename Value_, typename Index_, typename Right_, typename GetOutput_>
void multiply_dense_column_with_some_vectors_unblocked(
    const Index_ NR,
    const Index_ start,
    const Index_ length,
    const tatami::Matrix<Value_, Index_>& matrix,
    const std::size_t num_rhs,
    const std::vector<Right_*>& rhs_ptrs,
    GetOutput_ get_output
) {
    auto buffer = tatami::create_container_of_Index_size<std::vector<Output_> >(NR);
    auto ext = tatami::consecutive_extractor<false>(matrix, false, start, length);
    for (Index_ c = 0; c < length; ++c) {
        const auto ptr = ext->fetch(buffer.data());
        for (std::size_t j = 0; j < num_rhs; ++j) {
            auto&& optr = get_output(j);
            const Output_ mult = rhs_ptrs[j][start + c];
            for (Index_ r = 0; r < NR; ++r) {
                optr[r] += mult * static_cast<Output_>(ptr[r]);
            }
        }
    }
}

template<typename Output_, typename Value_, typename Index_, typename Right_, typename GetOutput_>
void multiply_dense_column_with_some_vectors_blocked(
    const Index_ NR,
    const Index_ start,
    const Index_ length,
    const tatami::Matrix<Value_, Index_>& matrix,
    const std::size_t num_rhs,
    const std::vector<Right_*>& rhs_ptrs,
    GetOutput_ get_output,
    const Index_ primary_block_size,
    const Index_ secondary_block_size
) {
    std::vector<std::vector<Value_> > buffers;
    std::vector<const Value_*> ptrs;
    {
        const Index_ num_buffers = std::min(length, primary_block_size);
        buffers.reserve(num_buffers);
        for (I<decltype(num_buffers)> b = 0; b < num_buffers; ++b) {
            buffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<Value_> >(NR));
        }
        sanisizer::resize(ptrs, num_buffers);
    }

    auto ext = tatami::consecutive_extractor<false>(matrix, false, start, length);
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
                        const Output_ mult = rhs_ptrs[hcopy][start + c + ccounter];
                        auto&& outcol = get_output(hcopy);
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

template<typename Value_, typename Index_, typename Right_, typename GetOutput_>
void multiply_dense_column_with_some_vectors(
    const tatami::Matrix<Value_, Index_>& matrix,
    const std::vector<Right_*>& rhs_ptrs,
    GetOutput_ get_output,
    const MultiplyDenseColumnWithSomeVectorsOptions& options
) {
    const Index_ NR = matrix.nrow();
    const Index_ NC = matrix.ncol();
    const auto num_rhs = rhs_ptrs.size();

    typedef I<decltype(get_output(0)[0])> Output_;
    for (std::size_t c = 0; c < num_rhs; ++c) {
        std::fill_n(get_output(c), NR, 0);
    }

    const bool do_parallel = options.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<std::vector<Output_> > > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(options.num_threads - 1));
    }

    int num_used;
    if (options.primary_block_size == 1) {
        num_used = tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
            if (!do_parallel || t == 0) {
                multiply_dense_column_with_some_vectors_unblocked<Output_>(
                    NR,
                    start,
                    length,
                    matrix,
                    num_rhs,
                    rhs_ptrs,
                    get_output
                );
                return;
            }

            auto tmp_out = sanisizer::create<std::vector<std::vector<Output_> > >(num_rhs);
            for (I<decltype(num_rhs)> c = 0; c < num_rhs; ++c) {
                tatami::resize_container_to_Index_size(tmp_out[c], NR);
            }

            multiply_dense_column_with_some_vectors_unblocked<Output_>(
                NR,
                start,
                length,
                matrix,
                num_rhs,
                rhs_ptrs,
                [&](const std::size_t r) -> std::vector<Output_>& { return tmp_out[r]; }
            );

            if (do_parallel && t > 0) {
                (*tmp_results)[t - 1] = std::move(tmp_out);
            }
        }, NC, options.num_threads);

    } else {
        const auto primary_block_size = sanisizer::cast<Index_>(options.primary_block_size);
        const auto secondary_block_size = sanisizer::cast<Index_>(options.secondary_block_size);
        num_used = tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
            if (!do_parallel || t == 0) {
                multiply_dense_column_with_some_vectors_blocked<Output_>(
                    NR,
                    start,
                    length,
                    matrix,
                    num_rhs,
                    rhs_ptrs,
                    get_output,
                    primary_block_size,
                    secondary_block_size
                );
                return;
            }

            auto tmp_out = sanisizer::create<std::vector<std::vector<Output_> > >(num_rhs);
            for (I<decltype(num_rhs)> c = 0; c < num_rhs; ++c) {
                tatami::resize_container_to_Index_size(tmp_out[c], NR);
            }

            multiply_dense_column_with_some_vectors_blocked<Output_>(
                NR,
                start,
                length,
                matrix,
                num_rhs,
                rhs_ptrs,
                [&](const std::size_t r) -> std::vector<Output_>& { return tmp_out[r]; },
                primary_block_size,
                secondary_block_size
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
