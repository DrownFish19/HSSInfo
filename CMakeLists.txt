cmake_minimum_required(VERSION 3.21)

project(HSSInfo LANGUAGES CXX CUDA)

# download CPM.cmake
set(CPM_DOWNLOAD_VERSION 0.34.0)
if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD
       https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
       ${CPM_DOWNLOAD_LOCATION}
  )
endif()

# include 
include(${CPM_DOWNLOAD_LOCATION})
CPMAddPackage(
    NAME CCCL
    GITHUB_REPOSITORY nvidia/cccl
    GIT_TAG v2.2.0
)

add_executable(HSSInfo
        main.cu
        hssinfo.cu)
target_link_libraries(HSSInfo PRIVATE CCCL::CCCL)