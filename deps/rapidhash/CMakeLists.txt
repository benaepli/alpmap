add_library(rapidhash INTERFACE)
add_library(rapidhash::rapidhash ALIAS rapidhash)

target_include_directories(rapidhash INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
