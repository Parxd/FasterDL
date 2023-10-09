cmake_minimum_required(VERSION 3.22)
project(main CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(main src/main.cu)

set_target_properties(main PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(main -lcublas)