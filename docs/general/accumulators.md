# Multiple accumulators {#multiple-accumulators}

We can use multiple accumulators to speed up the calculation of dot products.
That is, instead of computing the dot product as:

```cpp
double dot = 0;
for (std::size_t i = 0; i < N; ++i) {
    dot += left[i] * right[i];
}
```

We could instead use two accumulators:

```cpp
double dot1 = 0, dot2 = 0;
const std::size_t cycles = N / 2; 
for (std::size_t c = 0; c < cycles; ++c) {
    const auto i = 2 * c;
    dot1 += left[i] * right[i];
    dot2 += left[i + 1] * right[i + 1];
}
if (N % 2 == 1) {
    dot1 += left[N - 1] * right[N - 1];
}
double dot = dot1 + dot2;
```

The idea is to break dependency chains in the CPU's instruction pipeline by allowing multiple summations to occur in parallel.
It also provides some opportunities for auto-vectorization of the sum in the innermost loop.
However, we don't want to use too many accumulators as this could cause register spills and inflate the program size. 
Different numbers of accumulators may also slightly alter the output due to changes in floating-point round-off error.

The number of accumulators should be positive.
(Setting it to 1 will just fall back to the obvious summation.)
If multiple accumulators are to be used, we recommend setting the number of accumulators to a power of 2. 
This reduces the cost of the division to compute the number of loop iterations and is most compatible with vector instructions (if available).
Some testing indicates that 4 accumulators is a decent default, at least on Intel.
