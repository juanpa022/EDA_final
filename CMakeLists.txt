cmake_minimum_required(VERSION 3.14)
project(ivf_hnsw CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

if(MSVC)
    add_compile_options(/O2 /W3 /openmp)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra -fopenmp)
endif()

add_library(ivf_hnsw_lib STATIC
    ivf_hnsw/hnsw.cpp
    ivf_hnsw/ivf_hnsw.cpp
)
target_include_directories(ivf_hnsw_lib PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(ivf_hnsw_lib PUBLIC OpenMP::OpenMP_CXX)

add_executable(demo main.cpp)
target_link_libraries(demo PRIVATE ivf_hnsw_lib)
