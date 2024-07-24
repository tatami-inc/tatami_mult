#ifndef TATAMI_MULT_HPP
#define TATAMI_MULT_HPP

#include "tatami/tatami.hpp"

#include "dense_row.hpp"
#include "sparse_row.hpp"
#include "dense_column.hpp"
#include "sparse_column.hpp"

namespace tatami_mult {

struct Options {
    int num_threads = 1;

    bool prefer_largest = true;

    bool column_major_output = true;
};

template<typename Value_, typename Index_, typename Right_, typename Output_>
void multiply(const tatami::Matrix<Value_, Index_>& left, const Right_* right, Output_* output, const Options& opt) {
    if (left.sparse()) {
        if (left.prefer_rows()) {
            internal::sparse_row_vector(left, right, output, opt.num_threads);
        } else {
            internal::sparse_column_vector(left, right, output, opt.num_threads);
        }
    } else {
        if (left.prefer_rows()) {
            internal::dense_row_vector(left, right, output, opt.num_threads);
        } else {
            internal::dense_column_vector(left, right, output, opt.num_threads);
        }
    }
}

template<typename Left_, typename Value_, typename Index_, typename Output_>
void multiply(const Left_* left, const tatami::Matrix<Value_, Index_>& right, Output_* output, const Options& opt) {
    auto tright = tatami::wrap_shared_ptr(&right);
    if (tright->sparse()) {
        if (tright->prefer_rows()) {
            internal::sparse_row_vector(*tright, left, output, opt.num_threads);
        } else {
            internal::sparse_column_vector(*tright, left, output, opt.num_threads);
        }
    } else {
        if (tright->prefer_rows()) {
            internal::dense_row_vector(*tright, left, output, opt.num_threads);
        } else {
            internal::dense_column_vector(*tright, left, output, opt.num_threads);
        }
    }
}

template<typename Value_, typename Index_, typename Right_, typename Output_>
void multiply(const tatami::Matrix<Value_, Index_>& left, const std::vector<Right_*>& right, const std::vector<Output_*>& output, const Options& opt) {
    if (left.sparse()) {
        if (left.prefer_rows()) {
            internal::sparse_row_vectors(left, right, output, opt.num_threads);
        } else {
            internal::sparse_column_vectors(left, right, output, opt.num_threads);
        }
    } else {
        if (left.prefer_rows()) {
            internal::dense_row_vectors(left, right, output, opt.num_threads);
        } else {
            internal::dense_column_vectors(left, right, output, opt.num_threads);
        }
    }
}

template<typename Left_, typename Value_, typename Index_, typename Output_>
void multiply(const std::vector<Left_*>& left, const tatami::Matrix<Value_, Index_>& right, const std::vector<Output_*>& output, const Options& opt) {
    auto tright = tatami::wrap_shared_ptr(&right);
    if (tright->sparse()) {
        if (tright->prefer_rows()) {
            internal::sparse_row_vectors(*tright, left, output, opt.num_threads);
        } else {
            internal::sparse_column_vectors(*tright, left, output, opt.num_threads);
        }
    } else {
        if (tright->prefer_rows()) {
            internal::dense_row_vectors(*tright, left, output, opt.num_threads);
        } else {
            internal::dense_column_vectors(*tright, left, output, opt.num_threads);
        }
    }
}

/**
 * @cond
 */
namespace internal {

template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply(const tatami::Matrix<LeftValue_, LeftIndex_>& left, const tatami::Matrix<RightValue_, RightIndex_>& right, Output_* output, bool column_major_out, int num_threads) {
    size_t row_shift, col_shift;
    if (column_major_out) {
        row_shift = 1;
        col_shift = left.nrow();
    } else {
        row_shift = right.ncol();
        col_shift = 1;
    }

    if (left.sparse()) {
        if (left.prefer_rows()) {
            internal::sparse_row_matrix(left, right, output, row_shift, col_shift, opt.num_threads);
        } else {
            internal::sparse_column_matrix(left, right, output, row_shift, col_shift, opt.num_threads);
        }
    } else {
        if (tmatrix->prefer_rows()) {
            internal::dense_row_matrix(left, right, output, row_shift, col_shift, opt.num_threads);
        } else {
            internal::dense_column_matrix(left, right, output, row_shift, col_shift, opt.num_threads);
        }
    }
}

}
/**
 * @endcond
 */

template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply(const tatami::Matrix<LeftValue_, LeftIndex_>& left, const tatami::Matrix<RightValue_, RightIndex_>& right, Output_* output, const Options& opt) {
    // Pick the primary matrix.
    if (opt.prefer_largest) {
        if (left.nrow() < right.ncol()) {
            auto tright = tatami::wrap_shared_ptr(&right);
            auto tleft = tatami::wrap_shared_ptr(&left);
            internal::multiply(*tright, *tleft, output, !opt.column_major_output, opt.num_threads);
            return;
        }
    }

    internal::multiply(left, right, output, opt.column_major_output, opt.num_threads);
}

}

#endif
