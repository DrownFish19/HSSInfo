cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

set(CUB_DIR "/usr/local/cuda/targets/x86_64-linux/lib/cmake/cub")
set(Thrust_DIR "/usr/local/cuda/targets/x86_64-linux/lib/cmake/thrust")
set(THRUST_ENABLE_COMPUTE_80 ON)
set(THRUST_INCLUDE_CUB_CMAKE ON)
set(THRUST_INSTALL_CUB_HEADERS ON)
set(THRUST_ENABLE_COMPUTE_FUTURE ON)

# for Linux CUDA (only)
project(HSSInfo LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(CUB REQUIRED CONFIG)
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(ThrustCUDA HOST CPP DEVICE CUDA)

add_executable(HSSInfo
        main.cu
        hssinfo.cu)
target_link_libraries(HSSInfo ThrustCUDA)