# Matrix multiplication for tatami

![Unit tests](https://github.com/tatami-inc/tatami_mult/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/tatami-inc/tatami_mult/actions/workflows/doxygenate.yaml/badge.svg)
[![Codecov](https://codecov.io/gh/tatami-inc/tatami_mult/branch/master/graph/badge.svg?token=dwGIU5outm)](https://codecov.io/gh/tatami-inc/tatami_mult)

## Overview

This library contains functions for multiplying a [**tatami**](https://github.com/tatami-inc/tatami) matrix with a vector or matrix.
It provides specialized code paths depending on the properties of the `tatami::Matrix` - namely, sparsity or row-based iteration.
Parallelization is achieved via the usual `tatami::parallelize()` function.

## Quick start

Matrix-vector multiplication is pretty straightforward:

```cpp
#include "tatami_mult/tatami_mult.hpp"

std::shared_ptr<tatami::NumericMatrix> mat(
    new tatami::DenseRowMatrix<double, int>(nrows, ncols, vals)
);

// Create an RHS vector.
std::vector<double> rhs(mat.ncol());

// Store the matrix-vector product in 'output'.
std::vector<double> output(mat.nrow());
tatami_mult::Options opt;
tatami_mult::multiply(*mat, rhs.data(), output.data(), opt);
```

If multiple vectors are present, we can handle it in a single pass through our matrix:

```cpp
// Create multiple LHS vectors.
std::vector<std::vector<double> > lhs(10, std::vector<double>(mat.nrow()));
std::vector<const double*> lhs_ptrs(10);
for (size_t l = 0; l < lhs.size(); ++l) {
    lhs_ptrs[l] = lhs[l].data()
}

// Vector-matrix product for each vector.
std::vector<std::vector<double> > output(10, std::vector<double>(mat.ncol()));
std::vector<double*> out_ptrs(10);
for (size_t l = 0; l < output.size(); ++l) {
    out_ptrs[l] = output[l].data()
}

tatami_mult::Options opt;
tatami_mult::multiply(lhs_ptrs, *mat, out_ptrs, opt);
```

With two `tatami::Matrix` objects, `multiply()` will prefer a single pass through the larger matrix,
and will save the product as a column-major array.
Both of these behaviors can be modified by changing the settings in `Options`.

```cpp
std::shared_ptr<tatami::NumericMatrix> mat2(
    new tatami::DenseRowMatrix<double, int>(ncols, 100, vals)
);

std::vector<double> output(nrow, 100);
tatami_mult::multiply(mat, mat2, output.data(), opt);
```

Check out the [reference documentation](https://tatami-inc.github.io/tatami_mult) for more details.

## Building projects 

### CMake with `FetchContent`

If you're using CMake, you just need to add something like this to your `CMakeLists.txt`:

```cmake
include(FetchContent)

FetchContent_Declare(
  tatami_mult
  GIT_REPOSITORY https://github.com/tatami-inc/tatami_mult
  GIT_TAG master # or any version of interest 
)

FetchContent_MakeAvailable(tatami_mult)
```

Then you can link to **tatami_mult** to make the headers available during compilation:

```cmake
# For executables:
target_link_libraries(myexe tatami_mult)

# For libaries
target_link_libraries(mylib INTERFACE tatami_mult)
```

### CMake with `find_package()`

You can install the library by cloning a suitable version of this repository and running the following commands:

```sh
mkdir build && cd build
cmake .. -DTATAMI_MULT_TESTS=OFF
cmake --build . --target install
```

Then you can use `find_package()` as usual:

```cmake
find_package(tatami_tatami_mult CONFIG REQUIRED)
target_link_libraries(mylib INTERFACE tatami::tatami_mult)
```

### Manual

If you're not using CMake, the simple approach is to just copy the files the `include/` subdirectory - 
either directly or with Git submodules - and include their path during compilation with, e.g., GCC's `-I`.
You'll need to include the transitive dependencies yourself,
check out [`extern/CMakeLists.txt`](extern/CMakeLists.txt) for a list.
