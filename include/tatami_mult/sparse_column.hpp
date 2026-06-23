#ifndef TATAMI_MULT_SPARSE_COLUMN_HPP
#define TATAMI_MULT_SPARSE_COLUMN_HPP

#include "utils.hpp"

#include <vector>
#include <cstddef>
#include <type_traits>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

namespace tatami_mult {

template<typename Value_, typename Index_, typename Right_, typename Output_>
void sparse_column_vector(const tatami::Matrix<Value_, Index_>& matrix, const Right_* rhs, Output_* output, int num_threads) {
    const Index_ NR = matrix.nrow();
    const Index_ NC = matrix.ncol();

    const bool do_parallel = num_threads > 1; 
    std::optional<std::vector<std::optional<std::vector<Output_> > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(num_threads - 1));
    }
    std::fill_n(output, NR, 0);

    const auto num_used = tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(&matrix, false, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NR);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NR);

        Output_* optr;
        std::optional<std::vector<Output_> > cur_output;
        if (!do_parallel || t == 0) {
            optr = output;
        } else {
            cur_output.emplace(tatami::cast_Index_to_container_size<I<decltype(*cur_output)> >(NR));
            optr = cur_output->data();
        }

        for (Index_ c = 0; c < length; ++c) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());
            const Output_ mult = rhs[start + c];
            for (Index_ r = 0; r < range.number; ++r) {
                optr[range.index[r]] += mult * range.value[r];
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

template<typename Value_, typename Index_, typename Right_, typename GetOutput_>
void sparse_column_dense_vectors_internal(
    const Index_ start,
    const Index_ length,
    tatami::OracularSparseExtractor<Value_, Index_>& ext, 
    std::vector<Value_>& vbuffer,
    std::vector<Index_>& ibuffer,
    const std::size_t num_rhs,
    const bool rhs_columnar,
    const std::vector<Right_*>& rhs_ptrs,
    const bool output_columnar,
    GetOutput_ get_output
) {
    for (Index_ c = 0; c < length; ++c) {
        const auto range = ext.fetch(vbuffer.data(), ibuffer.data());

        if (!rhs_columnar) {
            const auto rptr = rhs_ptrs[start + c];
            if (output_columnar) {
                for (I<decltype(num_rhs)> j = 0; j < num_rhs; ++j) {
                    auto&& optr = get_output(j);
                    const auto mult = rptr[j];
                    for (Index_ x = 0; x < range.number; ++x) {
                        optr[range.index[x]] += mult * range.value[x];
                    }
                }

            } else {
                for (Index_ x = 0; x < range.number; ++x) {
                    auto&& optr = get_output(range.index[x]);
                    const auto mult = range.value[x];
                    for (I<decltype(num_rhs)> j = 0; j < num_rhs; ++j) {
                        optr[j] += mult * rptr[j];
                    }
                }
            }

        } else {
            if (output_columnar) {
                for (I<decltype(num_rhs)> j = 0; j < num_rhs; ++j) {
                    auto&& optr = get_output(j);
                    const auto mult = rhs_ptrs[j][start + c];
                    for (Index_ x = 0; x < range.number; ++x) {
                        optr[range.index[x]] += mult * range.value[x]; 
                    }
                }

            } else {
                // TODO: This is pretty gross, lots of non-contiguous access in the hottest loop.
                for (Index_ x = 0; x < range.number; ++x) {
                    auto&& optr = get_output(range.index[x]);
                    const auto mult = range.value[x];
                    for (I<decltype(num_rhs)> j = 0; j < num_rhs; ++j) {
                        optr[j] += mult * rhs_ptrs[j][start + c];
                    }
                }
            }
        }
    }
}

template<typename Value_, typename Index_, typename Right_, typename GetOutput_>
void sparse_column_dense_vectors(
    const tatami::Matrix<Value_, Index_>& matrix,
    const std::size_t num_rhs,
    const bool rhs_columnar,
    const std::vector<Right_*>& rhs_ptrs,
    const bool output_columnar,
    GetOutput_ get_output,
    const int num_threads
) {
    const Index_ NR = matrix.nrow();
    const Index_ NC = matrix.ncol();

    typedef I<decltype(get_output(0)[0])> Output_;
    if (output_columnar) {
        for (std::size_t c = 0; c < num_rhs; ++c) {
            std::fill_n(get_output(c), NR, 0);
        }
    } else {
        for (Index_ r = 0; r < NR; ++r) {
            std::fill_n(get_output(r), num_rhs, 0);
        }
    }

    const bool do_parallel = num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<std::vector<Output_> > > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(num_threads - 1));
    }

    const auto num_used = tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(matrix, false, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NR);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NR);

        if (!do_parallel || t == 0) {
            sparse_column_dense_vectors_internal(
                start,
                length,
                *ext,
                vbuffer,
                ibuffer,
                num_rhs,
                rhs_columnar,
                rhs_ptrs,
                output_columnar,
                get_output
            );
            return;
        }

        auto tmp_out = sanisizer::create<std::vector<std::vector<Output_> > >(output_columnar ? num_rhs : NR);
        if (output_columnar) {
            for (I<decltype(num_rhs)> c = 0; c < num_rhs; ++c) {
                tatami::resize_container_to_Index_size(tmp_out[c], NR);
            }
        } else {
            for (I<decltype(NR)> r = 0; r < NR; ++r) {
                tatami::resize_container_to_Index_size(tmp_out[r], num_rhs);
            }
        }

        sparse_column_dense_vectors_internal(
            start,
            length,
            *ext,
            vbuffer,
            ibuffer,
            num_rhs,
            rhs_columnar,
            rhs_ptrs,
            output_columnar,
            [&](const std::size_t r) -> std::vector<Output_>& { return tmp_out[r]; }
        );

        if (do_parallel && t > 0) {
            (*tmp_results)[t - 1] = std::move(tmp_out);
        }
    }, NC, num_threads);

    if (do_parallel) {
        for (int u = 1; u < num_used; ++u) {
            const auto& tmp = *((*tmp_results)[u - 1]);
            if (output_columnar) {
                for (I<decltype(num_rhs)> j = 0; j < num_rhs; ++j) {
                    const auto tmpvec = tmp[j];
                    const auto outptr = get_output(j);
                    for (Index_ r = 0; r < NR; ++r) {
                        outptr[r] += tmpvec[r];
                    }
                }
            } else {
                for (Index_ r = 0; r < NR; ++r) {
                    const auto tmpvec = tmp[r];
                    const auto outptr = get_output(r);
                    for (I<decltype(num_rhs)> j = 0; j < num_rhs; ++j) {
                        outptr[j] += tmpvec[j];
                    }
                }
            }
        }
    }
}

template<typename Value_, typename Index_, typename Right_, typename Output_>
void sparse_column_vectors(const tatami::Matrix<Value_, Index_>& matrix, const std::vector<Right_*>& rhs_ptrs, const std::vector<Output_*>& output, int num_threads) {
    sparse_column_dense_vectors(
        matrix,
        rhs_ptrs.size(),
        true,
        rhs_ptrs,
        true,
        [&](const std::size_t r) -> Output_* { return output[r]; },
        num_threads
    );
}

template<typename Value_, typename Index_, typename RightValue_, typename RightIndex_, typename Output_>
void sparse_column_tatami_dense(
    const tatami::Matrix<Value_, Index_>& matrix,
    const tatami::Matrix<RightValue_, RightIndex_>& rhs,
    const bool output_columnar,
    Output_* const output,
    int num_threads
) {
    const auto rhs_row = rhs.nrow();
    const auto num_rhs = rhs.ncol();

    // The general strategy here is to drag the (hopefully small-ish) RHS matrix into memory,
    // avoiding the need to repeatedly use tatami to iterate over it. 

    if (rhs.prefer_rows() || !output_columnar) {
        // For column-major RHS matrices, we forcibly extract rows if the output is not columnar.
        // This avoids a suboptimal extraction within sparse_column_dense_vectors(), see the comments above.
        // we consider a one-time suboptimal extraction to be cheaper than many suboptimal accesses for multiplication itself.
        auto all_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(rhs_row);
        auto all_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(rhs_row);
        populate_dense_buffers(true, rhs_row, num_rhs, rhs, all_buffers, all_ptrs, num_threads);

        if (output_columnar) {
            const auto NR = matrix.nrow();
            sparse_column_dense_vectors(
                matrix,
                num_rhs,
                false,
                all_ptrs,
                true,
                [&](const std::size_t c) -> Output_* { return output + sanisizer::product_unsafe<std::size_t>(c, NR); },
                num_threads
            );

        } else {
            sparse_column_dense_vectors(
                matrix,
                num_rhs,
                false,
                all_ptrs,
                false,
                [&](const std::size_t r) -> Output_* { return output + sanisizer::product_unsafe<std::size_t>(r, num_rhs); },
                num_threads
            );
        }

    } else {
        auto all_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(num_rhs);
        auto all_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(num_rhs);
        populate_dense_buffers(false, num_rhs, rhs_row, rhs, all_buffers, all_ptrs, num_threads);

        assert(output_columnar);
        const auto NR = matrix.nrow();
        sparse_column_dense_vectors(
            matrix,
            num_rhs,
            true,
            all_ptrs,
            true,
            [&](const std::size_t c) -> Output_* { return output + sanisizer::product_unsafe<std::size_t>(c, NR); },
            num_threads
        );
    }
}

template<typename Value_, typename Index_, typename RightValue_, typename RightIndex_, typename Output_>
void sparse_column_tatami_sparse(
    const tatami::Matrix<Value_, Index_>& matrix,
    const tatami::Matrix<RightValue_, RightIndex_>& rhs,
    const bool output_columnar,
    Output_* const output,
    int num_threads
) {
    const Index_ NR = matrix.nrow();
    const Index_ NC = matrix.ncol();
    assert(sanisizer::is_equal(NC, rhs.nrow()));
    const auto num_rhs = rhs.ncol();

    // The general strategy here is to drag the (hopefully small-ish) RHS matrix into memory,
    // avoiding the need to repeatedly use tatami to iterate over it. 
    // We must have a row-major layout for this to work, we can't rely on blocking for a sparse RHS.

    auto all_vbuffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(NC);
    auto all_ibuffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightIndex_> > >(NC);
    auto all_ranges = tatami::create_container_of_Index_size<std::vector<tatami::SparseRange<RightValue_, RightIndex_> > >(NC);
    populate_sparse_buffers(true, NC, num_rhs, rhs, all_vbuffers, all_ibuffers, all_ranges, num_threads);

    const auto full_size_output = sanisizer::product_unsafe<std::size_t>(NR, num_rhs);
    std::fill_n(output, full_size_output, 0);

    const bool do_parallel = num_threads > 1; 
    std::optional<std::vector<std::optional<std::vector<Output_> > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(num_threads - 1));
    }

    const auto num_used = tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(matrix, false, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NR);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NR);

        Output_* optr;
        std::optional<std::vector<Output_> > tmp_out;
        if (do_parallel && t != 0) {
            // If a full size array was allocated in 'output', we should be able to allocate a vector... hopefully.
            tmp_out.emplace(sanisizer::cast<typename std::vector<std::vector<Output_> >::size_type>(full_size_output));
            optr = tmp_out->data();
        } else {
            optr = output;
        }

        for (Index_ c = 0; c < length; ++c) {
            const auto mat_range = ext->fetch(vbuffer.data(), ibuffer.data());
            const auto& rhs_range = all_ranges[start + c];

            if (output_columnar) {
                for (I<decltype(rhs_range.number)> k = 0; k < rhs_range.number; ++k) {
                    const auto outcol = rhs_range.index[k];
                    const auto mult = rhs_range.value[k];
                    for (Index_ l = 0; l < mat_range.number; ++l) {
                        optr[sanisizer::nd_offset<std::size_t>(mat_range.index[l], NR, outcol)] += mult * mat_range.value[l];
                    }
                }
            } else {
                for (Index_ l = 0; l < mat_range.number; ++l) {
                    const auto outrow = mat_range.index[l];
                    const auto mult = mat_range.value[l];
                    for (I<decltype(rhs_range.number)> k = 0; k < rhs_range.number; ++k) {
                        optr[sanisizer::nd_offset<std::size_t>(rhs_range.index[k], num_rhs, outrow)] += mult * rhs_range.value[k];
                    }
                }
            }
        }

        if (do_parallel && t > 0) {
            (*tmp_results)[t - 1] = std::move(tmp_out);
        }
    }, NC, num_threads);

    if (do_parallel) {
        for (int u = 1; u < num_used; ++u) {
            const auto& tmp = *((*tmp_results)[u - 1]);
            for (std::size_t x = 0; x < full_size_output; ++x) {
                output[x] += tmp[x];
            }
        }
    }
}

}

#endif
