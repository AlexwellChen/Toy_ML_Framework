cmake_minimum_required(VERSION 3.22)
project(Toy_ML_Framework)

set(BREW_DIR /opt/homebrew/Cellar)

#set(BLAS_INC_DIR ${BREW_DIR}/openblas/0.3.17/include)
#set(BLAS_LINK_DIR ${BREW_DIR}/openblas/0.3.17/lib)
set(EIGEN_INC_DIR ${BREW_DIR}/eigen/3.3.9/include/eigen3)
set(IDIR include)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g")

add_executable(Toy_ML_Framework main.cpp include/MatOp.h src/MatAutoDiff.cpp include/MatAutoDiff.h include/MatAutoDiff.h example.cpp src/Model.cpp include/Model.h)

# 添加cblas
#include_directories (${BLAS_INC_DIR})
#link_directories(${BLAS_LINK_DIR})
#target_link_libraries(Toy_ML_Framework -lblas)

include_directories (${EIGEN_INC_DIR})
