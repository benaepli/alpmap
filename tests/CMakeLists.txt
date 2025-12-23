include(CTest)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(alpmap_test)

target_link_libraries(alpmap_test
        PRIVATE
        alpmap
        GTest::gtest
        GTest::gtest_main
)

target_sources(alpmap_test
    PRIVATE
        src/alpmap_test.cpp
)

add_test(NAME alpmap_test COMMAND alpmap_test)