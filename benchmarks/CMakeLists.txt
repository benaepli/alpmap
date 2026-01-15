include(FetchContent)
FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.4
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

FetchContent_Declare(
        absl
        GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
        GIT_TAG 20240116.2
)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ABSL_BUILD_MONOLITHIC_TESTS OFF CACHE BOOL "" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(absl)

add_executable(alpmap_benchmark
        src/common_benchmarks.cpp
        src/set_benchmark.cpp
        src/map_benchmark.cpp
)

target_link_libraries(alpmap_benchmark
        PRIVATE
        alpmap
        benchmark::benchmark_main
        absl::flat_hash_set
)

target_compile_features(alpmap_benchmark PRIVATE cxx_std_23)

target_compile_options(alpmap_benchmark PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>
)
