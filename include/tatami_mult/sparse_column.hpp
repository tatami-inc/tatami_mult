#ifndef TATAMI_MULT_SPARSE_COLUMN_HPP
#define TATAMI_MULT_SPARSE_COLUMN_HPP

#include <vector>
#include <cstdint>

#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"
#include "utils.hpp"

namespace tatami_mult {

namespace internal {

template<typename Value_, typename Index_, typename Output_>
void sparse_multiply_add(const tatami::SparseRange<Value_, Index_>& range, Index_ start, Output_ mult, Output_* optr) {
    for (Index_ r = 0; r < range.number; ++r) {
        optr[range.index[r] - start] += mult * range.value[r];
    }
}

template<typename Value_, typename Index_, typename Output_>
void sparse_expand_multiply_add(const tatami::SparseRange<Value_, Index_>& range, Index_ start, Index_ length, std::vector<Value_>& expanded, Output_ mult, Output_* optr) {
    expanded.resize(length);
    for (Index_ k = 0; k < range.number; ++k) {
        expanded[range.index[k] - start] = range.value[k];
    }
    for (Index_ r = 0; r < length; ++r) {
        optr[r] += expanded[r] * mult;
    }
    for (Index_ k = 0; k < range.number; ++k) {
        expanded[range.index[k] - start] = 0;
    }
}

template<typename Value_, typename Index_, typename Right_, typename Output_>
void sparse_column_vector(const tatami::Matrix<Value_, Index_>& matrix, const Right_* rhs, Output_* output, int num_threads) {
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();

    tatami::parallelize([&](size_t t, Index_ start, Index_ length) {
        auto ext = tatami::consecutive_extractor<true>(&matrix, false, 0, NC, start, length);
        std::vector<Value_> vbuffer(length);
        std::vector<Index_> ibuffer(length);

        tatami_stats::LocalOutputBuffer<Value_> store(t, start, length, output);
        auto optr = store.data();

        // Check if the RHS has any special values.
        constexpr bool supports_specials = supports_special_values<Right_>();
        typename std::conditional<supports_specials, std::vector<Value_>, bool>::type expanded;

        for (Index_ c = 0; c < NC; ++c) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());

            Output_ mult = rhs[c];
            if constexpr(supports_specials) {
                if (is_special(mult)) { // expanding it to explicitly perform the multiplication with specials.
                    sparse_expand_multiply_add(range, start, length, mult, optr);
                    continue;
                }
            }

            sparse_multiply_add(range, start, mult, optr);
        }

        store.transfer();
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename Right_, typename Output_>
void sparse_column_matrix(const tatami::Matrix<Value_, Index_>& matrix, const Right_* rhs, size_t rhs_col, Output_* output, int num_threads) {
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();

    tatami::parallelize([&](size_t t, Index_ start, Index_ length) {
        auto ext = tatami::consecutive_extractor<true>(&matrix, false, 0, NC, start, length);
        std::vector<Value_> vbuffer(length);
        std::vector<Index_> ibuffer(length);
        auto stores = create_stores(NR, rhs_col, start, length, output);

        for (Index_ c = 0; c < NC; ++c) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());

            size_t rhs_offset = c; // using offsets instead of directly adding the pointer, to avoid forming an invalid address on the final iteration.
            for (size_t j = 0; j < rhs_col; ++j, rhs_offset += NC) {
                auto optr = stores[j].data();
                Output_ mult = rhs[rhs_offset];

                if constexpr(supports_specials) {
                    if (is_special(mult)) { // expanding it to explicitly perform the multiplication with specials.
                        sparse_expand_multiply_add(range, start, length, mult, optr);
                        continue;
                    }
                }

                sparse_multiply_add(range, start, mult, optr);
            }
        }
   
        for (auto& s : stores) {
            s.transfer();
        }
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename RightValue_, typename RightIndex_, typename Output_>
void sparse_column_tatami_dense(const tatami::Matrix<Value_, Index_>& matrix, const tatami::Matrix<RightValue_, RightIndex_>& rhs, Output_* output, int num_threads) {
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();
    RightIndex_ rhs_col = rhs.ncol();

    tatami::parallelize([&](size_t t, Index_ start, Index_ length) {
        auto ext = tatami::consecutive_extractor<true>(&matrix, false, 0, NC, start, length);
        auto rext = tatami::consecutive_extractor<false>(&matrix, true, 0, NC); // remember, NC == rhs.nrow().
        std::vector<Value_> vbuffer(length);
        std::vector<Index_> ibuffer(length);
        std::vector<RightValue_> rbuffer(rhs_col);
        auto stores = create_stores(NR, rhs_col, start, length, output);

        for (Index_ c = 0; c < NC; ++c) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());
            auto rptr = rext->fetch(rbuffer.data());

            for (RightIndex_ j = 0; j < rhs_col; ++j) {
                auto optr = stores[j].data();
                Output_ mult = rptr[j];

                if constexpr(supports_specials) {
                    if (is_special(mult)) { // expanding it to explicitly perform the multiplication with specials.
                        sparse_expand_multiply_add(range, start, length, mult, optr);
                        continue;
                    }
                }

                sparse_multiply_add(range, start, mult, optr);
            }
        }
   
        for (auto& s : stores) {
            s.transfer();
        }
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename RightValue_, typename RightIndex_, typename Output_>
void sparse_column_tatami_sparse(const tatami::Matrix<Value_, Index_>& matrix, const tatami::Matrix<RightValue_, RightIndex_>& rhs, Output_* output, int num_threads) {
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();
    RightIndex_ rhs_col = rhs.ncol();

    tatami::parallelize([&](size_t, Index_ start, Index_ length) {
        auto ext = tatami::consecutive_extractor<true>(&matrix, false, 0, NC, start, length);
        auto rext = tatami::consecutive_extractor<true>(&matrix, true, 0, NC); // remember, NC == rhs.nrow().
        std::vector<Value_> vbuffer(length);
        std::vector<Index_> ibuffer(length);
        std::vector<RightValue_> rvbuffer(rhs_col);
        std::vector<RightIndex_> ribuffer(rhs_col);
        auto stores = create_stores(NR, rhs_col, start, length, output);

        constexpr bool supports_specials = supports_special_values<Value_>();
        typename std::conditional<supports_specials, std::vector<Index_>, bool>::type special_k;

        for (Index_ c = 0; c < NC; ++c) {
            auto range = ext->fetch(buffer.data(), ibuffer.data());
            auto rrange = rext->fetch(rvbuffer.data(), ribuffer.data());

            if constexpr(supports_specials) {
                specials.clear();
                for (Index_ k = 0; k < range.number; ++k) {
                    if (is_special(range.value[k])) {
                        special_k.push_back(k);
                    }
                }
            }

            if constexpr(supports_specials) { // need separate multiplication to preserve the specials.
                if (special_k.size()) {
                    RightIndex_ rk = 0; 
                    for (RightIndex_ j = 0; j < rhs_col; ++j) {
                        auto optr = stores[j].data();
                        if (rk < rrange.number && j == rrange.index[rk]) {
                            Output_ mult = rrange.value[rk];
                            for (Index_ k = 0; k < range.number; ++k) {
                                optr[range.index[k] - start] += mult * range.value[k];
                            }
                            ++rk;
                        } else {
                            for (auto k : special_k) {
                                optr[range.index[k] - start] += range.value[k] * static_cast<Output_>(0);
                            }
                        }
                    }
                    continue;
                }
            }

            for (RightIndex_ rk = 0; rk < range.number; ++rk) {
                auto optr = stores[rrange.index[rk]].data();
                Output_ mult = rrange.value[rk];
                for (Index_ k = 0; k < range.number; ++k) {
                    optr[range.index[k] - start] += mult * range.value[k];
                }
            }
        }
    }, NR, num_threads);
}

}

}

#endif
