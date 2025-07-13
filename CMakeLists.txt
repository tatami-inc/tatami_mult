cmake_minimum_required(VERSION 3.24)

project(tatami_mult
    VERSION 0.1.3
    DESCRIPTION "Multiplication of tatami matrices"
    LANGUAGES CXX)

add_library(tatami_mult INTERFACE)
add_library(tatami::tatami_mult ALIAS tatami_mult)

target_compile_features(tatami_mult INTERFACE cxx_std_17)

option(TATAMI_MULT_FETCH_EXTERN "Automatically fetch tatami_mult's external dependencies." ON)
if(TATAMI_MULT_FETCH_EXTERN)
    add_subdirectory(extern)
else()
    find_package(tatami_tatami 4.0.0 CONFIG REQUIRED)
    find_package(tatami_tatami_stats 2.0.1 CONFIG REQUIRED)
    find_package(ltla_sanisizer 0.1.0 CONFIG REQUIRED)
endif()

target_link_libraries(tatami_mult INTERFACE tatami::tatami tatami::tatami_stats ltla::sanisizer)

include(GNUInstallDirs)
target_include_directories(tatami_mult INTERFACE 
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/tatami_tatami_mult>"
)

# Building the test-related machinery, if we are compiling this library directly.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(TATAMI_MULT_TESTS "Build tatami_mult's test suite." ON)
else()
    option(TATAMI_MULT_TESTS "Build tatami_mult's test suite." OFF)
endif()

if(TATAMI_MULT_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif() 
endif()

# Installing for find_package.
include(CMakePackageConfigHelpers)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tatami_tatami_mult)

install(TARGETS tatami_mult
    EXPORT tatami_multTargets)

install(EXPORT tatami_multTargets
    FILE tatami_tatami_multTargets.cmake
    NAMESPACE tatami::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tatami_tatami_mult)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/tatami_tatami_multConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tatami_tatami_mult)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/tatami_tatami_multConfigVersion.cmake
    COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/tatami_tatami_multConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/tatami_tatami_multConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tatami_tatami_mult)
