include(FetchContent)
FetchContent_Declare(
  tatami_test
  GIT_REPOSITORY https://github.com/tatami-inc/tatami_test
  GIT_TAG master
)
FetchContent_MakeAvailable(tatami_test)

add_executable(
    libtest
    src/dense_dot_product.cpp
    src/sparse_dot_product.cpp
    src/single_vector/dense_row.cpp
    src/single_vector/dense_column.cpp
    src/single_vector/sparse_row.cpp
    src/single_vector/sparse_column.cpp
    src/multiple_vectors/dense_row.cpp
    src/multiple_vectors/dense_column/public.cpp
    src/multiple_vectors/sparse_row/public.cpp
    src/multiple_vectors/sparse_column/public.cpp
)

target_link_libraries(
    libtest
    tatami_mult
    tatami_test
)

target_compile_options(libtest PRIVATE -Wall -Wextra -Wpedantic -Werror)

set(CODE_COVERAGE OFF CACHE BOOL "Enable coverage testing")
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(libtest PRIVATE -O0 -g --coverage)
    target_link_options(libtest PRIVATE --coverage)
endif()

include(GoogleTest)
gtest_discover_tests(libtest)
