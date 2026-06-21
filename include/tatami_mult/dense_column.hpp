#ifndef TATAMI_MULT_DENSE_COLUMN_HPP
#define TATAMI_MULT_DENSE_COLUMN_HPP

#include "utils.hpp"

#include <vector>
#include <cstddef>
#include <type_traits>
#include <optional>
#include <algorithm>
#include <cassert>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

namespace tatami_mult {

template<typename Value_, typename Index_, typename Right_, typename Output_>
void dense_column_vector(const tatami::Matrix<Value_, Index_>& matrix, const Right_* rhs, Output_* output, int num_threads) {
    const Index_ NR = matrix.nrow();
    const Index_ NC = matrix.ncol();

    std::optional<std::vector<std::optional<std::vector<Output_> > > > tmp_results;
    const bool do_parallel = num_threads > 1;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(num_threads - 1));
    }
    std::fill_n(output, NR, 0);

    const auto num_used = tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<false>(&matrix, false, start, length);
        auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NR);

        Output_* optr;
        std::optional<std::vector<Output_> > cur_output;
        if (!do_parallel || t == 0) {
            optr = output;
        } else {
            cur_output.emplace(tatami::cast_Index_to_container_size<std::vector<Output_> >(NR));
            optr = cur_output->data();
        }

        for (Index_ c = 0; c < length; ++c) {
            auto ptr = ext->fetch(buffer.data());
            Output_ mult = rhs[start + c];
            for (Index_ r = 0; r < NR; ++r) {
                optr[r] += mult * ptr[r];
            }
        }

        if (do_parallel && t > 0) {
            (*tmp_results)[t - 1] = std::move(cur_output);            
        }
    }, NC, num_threads);

    if (do_parallel) {
        for (int u = 1; u < num_used; ++u) {
            const auto& tmp = *((*tmp_results)[u - 1]);
            for (Index_ r = 0; r < NR; ++r) {
                output[r] += tmp[r];
            }
        }
    }
}

template<typename Value_, typename Index_, typename Right_, typename Output_>
void dense_column_vectors(const tatami::Matrix<Value_, Index_>& matrix, const std::vector<Right_*>& rhs, const std::vector<Output_*>& output, int num_threads) {
    const Index_ NR = matrix.nrow();
    const Index_ NC = matrix.ncol();
    const auto num_rhs = rhs.size();

    const bool do_parallel = num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<std::vector<Output_> > > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(num_threads - 1));
    }
    for (const auto out : output) {
        std::fill_n(out, NR, 0);
    }

    const auto num_used = tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<false>(&matrix, false, start, length);
        auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NR);

        Output_* const * out_super_ptr;
        std::optional<std::vector<Output_*> > tmp_out_ptrs;
        std::optional<std::vector<std::vector<Output_> > > tmp_out_vecs;
        if (!do_parallel || t == 0) {
            out_super_ptr = output.data();
        } else {
            tmp_out_ptrs.emplace(sanisizer::cast<I<decltype(tmp_out_ptrs->size())> >(num_rhs));
            tmp_out_vecs.emplace(sanisizer::cast<I<decltype(tmp_out_vecs->size())> >(num_rhs));
            for (I<decltype(num_rhs)> r = 0; r < num_rhs; ++r) {
                tatami::resize_container_to_Index_size((*tmp_out_vecs)[r], NR);
                (*tmp_out_ptrs)[r] = (*tmp_out_vecs)[r].data();
            }
            out_super_ptr = tmp_out_ptrs->data();
        }

        for (Index_ c = 0; c < length; ++c) {
            const auto ptr = ext->fetch(buffer.data());
            for (I<decltype(num_rhs)> j = 0; j < num_rhs; ++j) {
                const Output_ mult = rhs[j][start + c];
                const auto optr = out_super_ptr[j];
                for (Index_ r = 0; r < NR; ++r) {
                    optr[r] += mult * ptr[r];
                }
            }
        }

        if (do_parallel && t > 0) {
            (*tmp_results)[t - 1] = std::move(tmp_out_vecs);
        }
    }, NC, num_threads);

    if (do_parallel) {
        for (int u = 1; u < num_used; ++u) {
            const auto& tmp = *((*tmp_results)[u - 1]);
            for (I<decltype(num_rhs)> j = 0; j < num_rhs; ++j) {
                const auto tmpvec = tmp[j];
                const auto outptr = output[j];
                for (Index_ r = 0; r < NR; ++r) {
                    outptr[r] += tmpvec[r];
                }
            }
        }
    }
}

template<typename Value_, typename Index_, typename RightValue_, typename RightIndex_, typename Output_>
void dense_column_tatami_dense(
    const tatami::Matrix<Value_, Index_>& matrix,
    const tatami::Matrix<RightValue_, RightIndex_>& rhs,
    Output_* output,
    RightIndex_ row_shift,
    Index_ col_shift,
    int num_threads
) {
    const Index_ NR = matrix.nrow();
    const Index_ NC = matrix.ncol();
    const RightIndex_ rhs_col = rhs.ncol();

    const bool do_parallel = num_threads > 1; 
    std::optional<std::vector<std::optional<std::vector<Output_> > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(num_threads - 1));
    }

    assert(row_shift == 1 || col_shift == 1);
    const auto output_size = sanisizer::product_unsafe<std::size_t>(NR, rhs_col);
    std::fill_n(output, output_size, 0);

    const auto num_used = tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<false>(&matrix, false, start, length);
        auto rext = tatami::consecutive_extractor<false, RightValue_, RightIndex_>(&rhs, true, start, length);

        auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NR);
        auto rbuffer = tatami::create_container_of_Index_size<std::vector<RightValue_> >(rhs_col);

        Output_* out_ptr;
        std::optional<std::vector<Output_> > cur_results;
        if (!do_parallel || t == 0) {
            out_ptr = output;            
        } else {
            cur_results.emplace(sanisizer::cast<I<decltype(cur_results->size())> >(output_size));
            out_ptr = cur_results->data();
        }

        if (row_shift == 1) { // i.e., column-major output.
            for (Index_ c = 0; c < length; ++c) {
                auto ptr = ext->fetch(buffer.data());
                auto rptr = rext->fetch(rbuffer.data());
                for (RightIndex_ j = 0; j < rhs_col; ++j) {
                    const auto out_col = out_ptr + sanisizer::product_unsafe<std::size_t>(j, NR);
                    const Output_ mult = rptr[j];
                    for (Index_ r = 0; r < NR; ++r) {
                        out_col[r] += mult * ptr[r];
                    }
                }
            }

        } else { // col_shift = 1, i.e., row-major output.
            for (Index_ c = 0; c < length; ++c) {
                auto ptr = ext->fetch(buffer.data());
                auto rptr = rext->fetch(rbuffer.data());
                for (Index_ r = 0; r < NR; ++r) {
                    const auto out_row = out_ptr + sanisizer::product_unsafe<std::size_t>(r, rhs_col);
                    const Output_ mult = ptr[r];
                    for (RightIndex_ j = 0; j < rhs_col; ++j) {
                        out_row[j] += mult * rptr[j];
                    }
                }
            }
        }

        if (do_parallel && t > 0) {
            (*tmp_results)[t - 1] = std::move(cur_results);
        }
    }, NC, num_threads);

    if (do_parallel) {
        for (int u = 1; u < num_used; ++u) {
            const auto& tmp = *((*tmp_results)[u - 1]);
            for (I<decltype(output_size)> x = 0; x < output_size; ++x) {
                output[x] += tmp[x];
            }
        }
    }
}

template<typename Value_, typename Index_, typename RightValue_, typename RightIndex_, typename Output_>
void dense_column_tatami_sparse(
    const tatami::Matrix<Value_, Index_>& matrix,
    const tatami::Matrix<RightValue_, RightIndex_>& rhs,
    Output_* output,
    RightIndex_ row_shift,
    Index_ col_shift,
    int num_threads
) {
    const Index_ NR = matrix.nrow();
    const Index_ NC = matrix.ncol();
    const RightIndex_ rhs_col = rhs.ncol();

    const bool do_parallel = num_threads > 1; 
    std::optional<std::vector<std::optional<std::vector<Output_> > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(num_threads - 1));
    }

    assert(row_shift == 1 || col_shift == 1);
    const auto output_size = sanisizer::product_unsafe<std::size_t>(NR, rhs_col);
    std::fill_n(output, output_size, 0);

    const auto num_used = tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<false>(&matrix, false, start, length);
        auto rext = tatami::consecutive_extractor<true>(&rhs, true, start, length);
        auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NR);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<RightValue_> >(rhs_col);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<RightIndex_> >(rhs_col);

        Output_* out_ptr;
        std::optional<std::vector<Output_> > cur_results;
        if (!do_parallel || t == 0) {
            out_ptr = output;            
        } else {
            cur_results.emplace(sanisizer::cast<I<decltype(cur_results->size())> >(output_size));
            out_ptr = cur_results->data();
        }

        if (row_shift == 1) { // i.e., column-major.
            for (Index_ c = 0; c < length; ++c) {
                auto ptr = ext->fetch(buffer.data());
                auto range = rext->fetch(vbuffer.data(), ibuffer.data());
                for (RightIndex_ k = 0; k < range.number; ++k) {
                    const auto out_col = out_ptr + sanisizer::product_unsafe<std::size_t>(range.index[k], NR);
                    const Output_ mult = range.value[k];
                    for (Index_ r = 0; r < NR; ++r) {
                        out_col[r] += mult * ptr[r];
                    }
                }
            }

        } else { // col_shift == 1, i.e., row-major.
            for (Index_ c = 0; c < length; ++c) {
                auto ptr = ext->fetch(buffer.data());
                auto range = rext->fetch(vbuffer.data(), ibuffer.data());
                for (Index_ r = 0; r < NR; ++r) {
                    const auto out_row = out_ptr + sanisizer::product_unsafe<std::size_t>(r, rhs_col);
                    const Output_ mult = ptr[r];
                    for (RightIndex_ k = 0; k < range.number; ++k) {
                        out_row[range.index[k]] += mult * range.value[k];
                    }
                }
            }
        }

        if (do_parallel && t > 0) {
            (*tmp_results)[t - 1] = std::move(cur_results);
        }
    }, NC, num_threads);

    if (do_parallel) {
        for (int u = 1; u < num_used; ++u) {
            const auto& tmp = *((*tmp_results)[u - 1]);
            for (I<decltype(output_size)> x = 0; x < output_size; ++x) {
                output[x] += tmp[x];
            }
        }
    }
}

}

#endif
