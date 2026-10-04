cmake_minimum_required(VERSION 3.12)
project(fast_ensemble)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# AVX2, O3 Optimizasyonu ve OpenMP Bayrakları
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp -ffast-math")

find_package(pybind11 REQUIRED)

pybind11_add_module(fast_ensemble ensemble_physics.cpp)

# OpenMP Kütüphanesini bağla
find_package(OpenMP REQUIRED)
target_link_libraries(fast_ensemble PRIVATE OpenMP::OpenMP_CXX)
