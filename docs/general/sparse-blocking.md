# Blocking for sparse matrices {#sparse-blocking}

Generally, the most expensive part of sparse matrix multiplication is the random accesses of the accompanying dense vector.
For example, when computing a dense-sparse dot product, we need to access the value of the dense vector at the index of each structural non-zero in the sparse vector.
Each access might require a fetch from main memory if the position of the structural non-zero falls in a new cache line.

To avoid this, we process a block of \f$B\f$ sparse vectors for each dense vector.
For example, we might load a block of sparse LHS rows and compute the dot product for each sparse row with a single dense RHS column.
This keeps the single dense vector in cache so that it can be quickly re-used for multiple sparse vectors.
We assume that each sparse vector in the block contains structural non-zeros at positions that lie within a cache line of the previous sparse vectors;
and that the cache hierarchy is large enough to hold the entire dense vector, or at least the parts corresponding to all of the structural non-zeros.
Larger values of \f$B\f$ improve speed at the cost of increased main memory usage as **tatami** needs to realize all sparse vectors into memory.

Additionally, if the dense vectors are small, we might consider loading a block of \f$C\f$ dense vectors into cache at once.
For each sparse vector, we iterate over the dense vectors in this block to perform the relevant multiplication operations.
We then repeat this with a new sparse vector but re-using the same block of dense vectors.
The idea is that the dense block is small enough that all of its vectors can be completely cached for quick re-use.
Obviously, this has pretty limited applicability with regards to the matrix shape.
