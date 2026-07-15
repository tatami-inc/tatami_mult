# Blocking for dense matrices {#dense-blocking}

## Overview 

When computing the product of two dense matrices, we can use blocking to improve cache utilization.
This involves splitting each matrix into smaller submatrices and computing the matrix product for pairs of submatrices.
Each of the submatrices should be small enough to be stored in L1 cache for fast re-use of rows/columns. 

To illustrate, consider the product of a row-major LHS matrix and a column-major RHS matrix.
Each LHS row is re-used to compute the dot product with each RHS column.
If the extent of the shared dimension is large enough to trigger cache evictions,
each LHS row or RHS column (depending on the iteration pattern) would need to be reloaded from memory on every use.
With blocking, we can reuse cached parts of multiple LHS rows with cached parts of multiple RHS columns to compute partial dot products.
We repeat this for each pair of submatrices and aggregate the results to obtain the full matrix product.

## Parameters

The exact blocking strategy depends on the layout of the the RHS, LHS and output matrices,
but we generally expect to operate on two \f$B\f$-by-\f$C\f$ (or \f$C\f$-by-\f$B\f$) matrices and one \f$B\f$-by-\f$B\f$ matrix:

- \f$B\f$ is the "primary" block size and determines the amount of cache re-use.
  In our example of a product of a row-major LHS and column-major RHS, each (part of a) LHS row only needs to be reloaded into cache once every \f$B\f$ RHS columns,
  compared to a naive approach where each LHS row may need to be reloaded into memory for each RHS column.
  Conversely, each (part of) an RHS column only needs to be loaded once every \f$B\f$ LHS rows.
- \f$C\f$ is the "secondary" block size and is the extent of the fastest-changing dimension.
  It should generally be the larger number to reduce the looping overhead.

In this framework, \f$2BC + B^2\f$ is the number of elements to be held in cache at any given time, plus sundries based on the granularity of the cache lines.
For a given cache size, a larger \f$B\f$ will improve cache re-use but increase the overhead from loop restarts due to a lower \f$C\f$.

The best choice of \f$B\f$ and \f$C\f$ depends on the size of the cache and the size of the data type.
If we're working with double-precision types, requiring \f$BC = 1024\f$ and enforcing \f$B \leq C\f$ will use 16-24 kb, which should easily fit into a typical 32 kb L1 cache.

## Further considerations 

Both \f$B\f$ and \f$C\f$ should be positive.

We recommend choosing a power of 2 for both \f$B\f$ and \f$C\f$ as this gives us a chance to exploit existing data alignment and vectorization. 
Indeed, blocking can be combined with @ref multiple-accumulators "multiple accumulators",
in which case \f$C\f$ should be a multiple of the number of accumulators to minimize entry into the epilogue loop.

A larger \f$B\f$ increases memory usage as more dimension elements need to be realized by **tatami**.

If the primary block size is set to 1 in any **tatami_mult** function, no blocking will be performed.
In this case, the choice of secondary block size will have no effect, i.e., \f$C\f$ is ignored.
