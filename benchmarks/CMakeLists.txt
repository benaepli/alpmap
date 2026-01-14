include(FetchContent)
FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(alpmap_benchmark
        src/common_benchmarks.cpp
        src/set_benchmark.cpp
        src/map_benchmark.cpp
)

target_link_libraries(alpmap_benchmark
        PRIVATE
        alpmap
        benchmark::benchmark_main
)

target_compile_features(alpmap_benchmark PRIVATE cxx_std_23)

target_compile_options(alpmap_benchmark PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>
)
