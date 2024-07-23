#ifndef TATAMI_MULT_DENSE_COLUMN_HPP
#define TATAMI_MULT_DENSE_COLUMN_HPP

#include <vector>

#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"
#include "utils.hpp"

namespace tatami_mult {

namespace internal {

template<typename Value_, typename Index_, typename Right_, typename Output_>
void dense_column_vector(const tatami::Matrix<Value_, Index_>& matrix, const Right_* rhs, Output_* output, int num_threads) {
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();

    tatami::parallelize([&](size_t t, Index_ start, Index_ length) {
        auto ext = tatami::consecutive_extractor<false>(&matrix, false, 0, NC, start, length);
        std::vector<Value_> buffer(NC);
        tatami_stats::LocalOutputBuffer<Value_> store(t, start, length, output);
        auto optr = store.data();

        for (Index_ c = 0; c < NC; ++c) {
            auto ptr = ext->fetch(buffer.data());
            Output_ mult = rhs[c];
            for (Index_ r = 0; r < length; ++r) {
                optr[r] += mult * ptr[r];
            }
        }

        store.transfer();
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename Right_, typename Output_>
void dense_column_matrix(const tatami::Matrix<Value_, Index_>& matrix, const Right_* rhs, size_t rhs_col, Output_* output, int num_threads) {
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();

    tatami::parallelize([&](size_t t, Index_ start, Index_ length) {
        auto ext = tatami::consecutive_extractor<false>(&matrix, false, 0, NC, start, length);
        std::vector<Value_> buffer(length);
        auto stores = create_stores(NR, rhs_col, start, length, output);

        for (Index_ c = 0; c < NC; ++c) {
            auto ptr = ext->fetch(buffer.data());

            size_t rhs_offset = c; // using offsets instead of directly adding the pointer, to avoid forming an invalid address on the final iteration.
            for (size_t j = 0; j < rhs_col; ++j, rhs_offset += NC) {
                auto optr = stores[j].data();
                Output_ mult = rhs[rhs_offset];
                for (Index_ r = 0; r < length; ++r) {
                    optr[r] += mult * ptr[r];
                }
            }
        }
   
        for (auto& s : stores) {
            s.transfer();
        }
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename RightValue_, typename RightIndex_, typename Output_>
void dense_column_tatami_dense(const tatami::Matrix<Value_, Index_>& matrix, const tatami::Matrix<RightValue_, RightIndex_>& rhs, Output_* output, int num_threads) {
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();
    RightIndex_ rhs_col = rhs.ncol();

    tatami::parallelize([&](size_t t, Index_ start, Index_ length) {
        auto ext = tatami::consecutive_extractor<false>(&matrix, false, 0, NC, start, length);
        auto rext = tatami::consecutive_extractor<false>(&matrix, true, 0, NC); // remember, NC == rhs.nrow().
        std::vector<Value_> buffer(length);
        std::vector<RightValue_> rbuffer(rhs_col);
        auto stores = create_stores(NR, rhs_col, start, length, output);

        for (Index_ c = 0; c < NC; ++c) {
            auto ptr = ext->fetch(buffer.data());
            auto rptr = rext->fetch(rbuffer.data());

            for (RightIndex_ j = 0; j < rhs_col; ++j) {
                auto optr = stores[j].data();
                Output_ mult = rptr[j];
                for (Index_ r = 0; r < length; ++r) {
                    optr[r] += mult * ptr[r];
                }
            }
        }
   
        for (auto& s : stores) {
            s.transfer();
        }
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename RightValue_, typename RightIndex_, typename Output_>
void dense_column_tatami_sparse(const tatami::Matrix<Value_, Index_>& matrix, const tatami::Matrix<RightValue_, RightIndex_>& rhs, Output_* output, int num_threads) {
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();
    RightIndex_ rhs_col = rhs.ncol();

    tatami::parallelize([&](size_t, Index_ start, Index_ length) {
        auto ext = tatami::consecutive_extractor<false>(&matrix, false, 0, NC, start, length);
        auto rext = tatami::consecutive_extractor<true>(&matrix, true, 0, NC); // remember, NC == rhs.nrow().
        std::vector<Value_> buffer(length);
        std::vector<RightValue_> vbuffer(rhs_col);
        std::vector<RightIndex_> ibuffer(rhs_col);
        auto stores = create_stores(NR, rhs_col, start, length, output);

        constexpr bool supports_specials = supports_special_values<Value_>();
        typename std::conditional<supports_specials, std::vector<Index_>, bool>::type specials;

        for (Index_ c = 0; c < NC; ++c) {
            auto ptr = ext->fetch(buffer.data());
            auto range = rext->fetch(vbuffer.data(), ibuffer.data());

            if constexpr(supports_specials) {
                specials.clear();
                fill_special_index(length, ptr, specials);
            }

            if constexpr(supports_specials) { // need separate multiplication to preserve the specials.
                if (specials.size()) {
                    RightIndex_ k = 0; 
                    for (RightIndex_ j = 0; j < rhs_col; ++j) {
                        auto optr = stores[j].data();
                        if (k < range.number && j == range.index[k]) {
                            Output_ mult = range.value[k];
                            for (Index_ r = 0; r < length; ++r) {
                                optr[r] += mult * ptr[r];
                            }
                            ++k;
                        } else {
                            for (auto s : specials) {
                                optr[s] += ptr[s] * static_cast<Output_>(0);
                            }
                        }
                    }
                    continue;
                }
            }

            for (RightIndex_ k = 0; k < range.number; ++k) {
                auto optr = stores[range.index[k]].data();
                Output_ mult = range.value[k];
                for (Index_ r = 0; r < length; ++r) {
                    optr[r] += mult * ptr[r];
                }
            }
        }
    }, NR, num_threads);
}

}

}

#endif