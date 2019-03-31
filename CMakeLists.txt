cmake_minimum_required(VERSION 3.13)
project(MachineLearning)

set(CMAKE_CXX_STANDARD 14)

include_directories(/mnt/sdb1/Project/MachineLearning/eigen/)

add_executable(MachineLearning Common.cpp Logistic.cpp MachineLearning.cpp Multi.cpp Single.cpp)
