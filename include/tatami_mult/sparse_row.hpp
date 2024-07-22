#ifndef TATAMI_MULT_SPARSE_ROW_HPP
#define TATAMI_MULT_SPARSE_ROW_HPP

#include <vector>
#include <cstdint>

#include "tatami/tatami.hpp"
#include "utils.hpp"

namespace tatami_mult {

namespace internal {

template<typename Value_, typename Index_, typename Right_, typename Output_>
void sparse_row_vector(const tatami::Matrix<Value_, Index_>& matrix, const Right_* rhs, Output_* output, int num_threads) {
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();

    // Check if the RHS has any special values.
    constexpr bool supports_specials = supports_special_values<Right_>();
    typename std::conditional<supports_specials, std::vector<Index_>, bool>::type specials;
    if constexpr(supports_specials) {
        fill_special_index(NC, rhs, specials);
    }

    tatami::parallelize([&](size_t, Index_ start, Index_ length) {
        auto ext = tatami::consecutive_extractor<true>(&matrix, true, start, length);
        std::vector<Value_> vbuffer(NC);
        std::vector<Index_> ibuffer(NC);

        for (Index_ r = start, end = start + length; r < end; ++r) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());

            if constexpr(supports_specials) {
                if (specials.size()) {
                    output[r] = special_dense_sparse_multiply<Output_>(specials, rhs, range);
                    continue;
                }
            }

            output[r] = dense_sparse_multiply<Output_>(rhs, range);
        }
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename Right_, typename Output_>
void sparse_row_matrix(const tatami::Matrix<Value_, Index_>& matrix, const Right_* rhs, size_t rhs_col, Output_* output, int num_threads) {
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();

    // Check if the RHS has any special values.
    constexpr bool supports_specials = supports_special_values<Right_>();
    typename std::conditional<supports_specials, std::vector<std::vector<Index_> >, bool>::type specials;
    if constexpr(supports_specials) {
        specials.resize(rhs_col);
        size_t rhs_offset = 0; // using offsets instead of directly adding to the pointer, to avoid forming an invalid address on the final iteration.
        for (size_t j = 0; j < rhs_col; ++j, rhs_offset += NC) {
            fill_special_index(NC, rhs + rhs_offset, specials[j]);
        }
    }

    tatami::parallelize([&](size_t, Index_ start, Index_ length) {
        auto ext = tatami::consecutive_extractor<true>(&matrix, true, start, length);
        std::vector<Value_> vbuffer(NC);
        std::vector<Index_> ibuffer(NC);

        for (Index_ r = start, end = start + length; r < end; ++r) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());
            size_t rhs_offset = 0;
            size_t out_offset = r; // using offsets instead of directly adding to the pointer, to avoid forming an invalid address on the final iteration.

            for (size_t j = 0; j < rhs_col; ++j, rhs_offset += NC, out_offset += NR) {
                if constexpr(supports_specials) {
                    if (specials[j].size()) {
                        output[out_offset] = special_dense_sparse_multiply<Output_>(specials[j], rhs + rhs_offset, range);
                        continue;
                    }
                }
                output[out_offset] = dense_sparse_multiply<Output_>(rhs + rhs_offset, range);
            }
        }
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename RightValue_, typename RightIndex_, typename Output_>
void sparse_row_tatami_dense(const tatami::Matrix<Value_, Index_>& matrix, const tatami::Matrix<RightValue_, RightIndex_>& rhs, Output_* output, int num_threads) {
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();
    RightIndex_ rhs_col = rhs.ncol();

    // Do one pass through the other matrix to see which columns have special values.
    // We can't afford to hold the indices here as rhs_col may be arbitrarily large
    // and the matrix might be full of special values.
    constexpr bool supports_specials = supports_special_values<RightValue_>();
    typename std::conditional<supports_specials, std::vector<uint8_t>, bool>::type has_special;
    bool any_special = false;
    if constexpr(supports_specials) {
        has_special.resize(rhs_col);

        tatami::parallelize([&](size_t, Index_ start, Index_ length) {
            auto rext = tatami::consecutive_extractor<false>(&rhs, false, start, length);
            std::vector<RightValue_> buffer(NC); // remember, NC == right.nrow() here.
            for (RightIndex_ c = start, end = start + length; c < end; ++c) {
                auto rptr = rext->fetch(buffer.data());
                for (RightIndex_ r = 0; r < NC; ++r) {
                    if (is_special(rptr[r])) {
                        has_special[r] = true;
                        break;
                    }
                }
            }
        }, rhs_col, num_threads);

        for (auto is : has_special) {
            if (is) {
                any_special = true;
                break;
            }
        }
    }

    tatami::parallelize([&](size_t, Index_ start, Index_ length) {
        auto ext = tatami::consecutive_extractor<true>(&matrix, true, start, length);
        std::vector<Value_> vbuffer(NC);
        std::vector<Index_> ibuffer(NC);
        std::vector<RightValue_> rbuffer(NC);

        // The idea is that if we have special values in the RHS, we expand the sparse
        // values into a dense array for a regular inner product. This avoids having to
        // keep track of the individual indices of the special values.
        typename std::conditional<supports_specials, std::vector<Value_>, bool>::type expanded;
        if constexpr(supports_specials) {
            if (any_special) {
                expanded.resize(NC);
            }
        }

        for (Index_ r = start, end = start + length; r < end; ++r) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());
            auto rext = tatami::consecutive_extractor<false>(&rhs, false, 0, rhs_col);
            size_t out_offset = r; // using offsets instead of directly adding to the pointer, to avoid forming an invalid address on the final iteration.

            if constexpr(supports_specials) {
                if (any_special) {
                    for (Index_ i = 0; i < range.number; ++i) {
                        expanded[range.index[i]] = range.value[i];
                    }

                    for (RightIndex_ j = 0; j < rhs_col; ++j, out_offset += NR) {
                        auto rptr = rext->fetch(rbuffer.data());
                        if (has_special[j]) {
                            output[out_offset] = std::inner_product(expanded.begin(), expanded.end(), rptr, static_cast<Output_>(0));
                        } else {
                            output[out_offset] = dense_sparse_multiply<Output_>(rptr, range);
                        }
                    }

                    for (Index_ i = 0; i < range.number; ++i) {
                        expanded[range.index[i]] = 0;
                    }
                    continue;
                }
            }

            for (RightIndex_ j = 0; j < rhs_col; ++j, out_offset += NR) {
                auto rptr = rext->fetch(rbuffer.data());
                output[out_offset] = dense_sparse_multiply<Output_>(rptr, range);
            }
        }
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename RightValue_, typename RightIndex_, typename Output_>
void sparse_row_tatami_sparse(const tatami::Matrix<Value_, Index_>& matrix, const tatami::Matrix<RightValue_, RightIndex_>& rhs, Output_* output, int num_threads) {
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();
    RightIndex_ rhs_col = rhs.ncol();

    tatami::parallelize([&](size_t, Index_ start, Index_ length) {
        auto ext = tatami::consecutive_extractor<true>(&matrix, true, start, length);
        std::vector<Value_> vbuffer(NC);
        std::vector<Index_> ibuffer(NC);
        std::vector<RightValue_> rvbuffer(NC);
        std::vector<RightIndex_> ribuffer(NC);
        std::vector<Value_> expanded;

        constexpr bool supports_specials = supports_special_values<RightValue_>();
        typename std::conditional<supports_specials, std::vector<Index_>, bool>::type specials;
        if constexpr(supports_specials) {
            expanded.resize(NC);
            specials.reserve(NC);
        }

        for (Index_ r = start, end = start + length; r < end; ++r) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());
            auto rext = tatami::consecutive_extractor<false>(&rhs, false, 0, rhs_col);

            // Expanding the sparse vector into a dense format for easier mapping by the RHS's sparse vector.
            for (Index_ i = 0; i < range.number; ++i) {
                expanded[range.index[i]] = range.value;
            }

            if constexpr(supports_specials) {
                specials.clear();
                for (Index_ i = 0; i < range.number; ++i) {
                    if (is_special(range.value[i])) {
                        specials.push_back(range.index[i]);
                    }
                }
            }

            size_t out_offset = r; // using offsets instead of directly adding to the pointer, to avoid forming an invalid address on the final iteration.
            for (RightIndex_ j = 0; j < rhs_col; ++j, out_offset += NR) {
                auto rrange = rext->fetch(vbuffer.data(), ibuffer.data());

                if constexpr(supports_specials) {
                    if (specials.size()) {
                        output[out_offset] = special_dense_sparse_multiply<Output_>(specials, expanded.data(), rrange);
                        continue;
                    }
                }

                output[out_offset] = dense_sparse_multiply<Output_>(expanded.data(), rrange);
            }

            for (Index_ i = 0; i < range.number; ++i) {
                expanded[range.index[i]] = 0;
            }
        }
    }, NR, num_threads);
}

}

}

#endif
