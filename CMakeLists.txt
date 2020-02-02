cmake_minimum_required(VERSION 3.13)
project(CatchTraining)

set(CMAKE_CXX_STANDARD 14)

include_directories(/mnt/d/Project/MachineLearning/eigen/)

add_executable(CatchTraining main.cpp Common.cpp Logistic.cpp Multi.cpp Single.cpp)
