cmake_minimum_required(VERSION 3.22)
project(tests)
set(CMAKE_CXX_STANDARD 17)

add_executable(main value.cpp)

target_sources(
    main PRIVATE
    ../../core/autodiff/graph/node.hpp
    ../../core/autodiff/graph/value.hpp
    )