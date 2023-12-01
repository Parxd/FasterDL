cmake_minimum_required(VERSION 3.22)
project(tests)
set(CMAKE_CXX_STANDARD 20)

add_executable(main value.cpp)

target_sources(
    main PRIVATE
    # ../../core/autodiff/value.cpp
    # ../../core/autodiff/add.cpp
    )