# Developer notes

## False sharing

False sharing can be ignored for algorithms that only involve intermittent writes to the shared output array.
More specifically, we're referring to algorithms involving a row-major LHS and a column-major RHS matrix (or equivalently, any number of RHS vectors),
where the dot product is computed for each LHS row/RHS column pair and stored in a single entry of the output array.

- As the writes do not occur in the innermost loop (i.e., the dot product), we'll assume that they're infrequent enough that false sharing won't be a problem. 
  This is reasonable if the common dimension extent is large enough that most time is spent computing the dot product.
  Any cache coherency-related stall should be easier to hide via out-of-order execution when the CPU starts computing the next dot product.
- We only have to worry about writes near the boundaries of the per-thread blocks.
  This means that false sharing will be even less of a problem if the number of LHS rows is large compared to the number of threads.
  For row-major output, it's effectively negligble as the per-thread blocks are contiguous such that there are very few boundaries.
- We're only ever writing to the output array, we never read from it.
  This provides even more opportunities for CPUs to mask the false sharing penalty, e.g., by collecting stores in a store buffer and to reduce the frequency of cache writes.
- We can mitigate false sharing by allocating a per-thread block of memory to store all results, and then transferring them to the output array in the serial section.
  However, this requires another allocation and an extra transfer step, which might not be faster than just putting up with a bit of false sharing.
  In particular, the transfer is particularly slow for column-major output where the results need to be interleaved across threads.

False sharing is more of a concern for algorithms that update the output array in the innermost loop, e.g., row-major RHS.
In such cases, we will allocate a per-thread block of memory to store the temporary results until all updates are complete.
(See https://github.com/tatami-inc/test-false_sharing, where per-thread allocations are pretty good at avoiding false sharing.) 
Then, once the final result is available, it is written once to the output array.
The risk of false sharing in this final write is no worse than the scenario described above where we deliberately ignore false sharing.
