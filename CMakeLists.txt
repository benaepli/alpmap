cmake_minimum_required(VERSION 3.30)

project(alpmap VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

# Build options
option(BUILD_TESTS "Build test programs" ON)
option(BUILD_BENCHMARKS "Build benchmark programs" OFF)
option(BUILD_DOCS "Build Doxygen documentation" ON)


include(FetchContent)

option(ALP_USE_EVE "Use EVE library for SIMD" ON)
if (ALP_USE_EVE)
    set(EVE_BUILD_DOCUMENTATION OFF)
    set(EVE_BUILD_BENCHMARKS OFF)
    set(EVE_BUILD_TEST OFF)
    set(EVE_BUILD_EXAMPLES OFF)

    FetchContent_Declare(
            eve
            URL https://github.com/jfalcou/eve/archive/refs/tags/v2023.02.15.zip
    )
    FetchContent_MakeAvailable(eve)
endif ()


add_library(alpmap
        src/backends/eve.hpp
        src/backends/sse.hpp)

if (ALP_USE_EVE)
    target_compile_definitions(alpmap PRIVATE ALP_USE_EVE)
    target_link_libraries(alpmap PRIVATE eve::eve)
endif ()

target_sources(alpmap
        PUBLIC
        FILE_SET CXX_MODULES FILES
        src/alp.cppm
        src/alp-map.cppm
        src/alp-set.cppm
        src/backends/std_simd.cppm
)

target_compile_features(alpmap PUBLIC cxx_std_23)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

if (BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif ()

if (BUILD_DOCS)
    add_subdirectory(docs)
endif ()