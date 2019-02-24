cmake_minimum_required(VERSION 3.13)
project(MachineLearning)

set(CMAKE_CXX_STANDARD 14)

include_directories(/mnt/sdb1/Project/MachineLearning/eigen/)

add_executable(
        Common.cpp
        Common.h
        Logistic.cpp
        Logistic.h
        MachineLearning.cpp
        Multi.cpp
        Multi.h
        Single.cpp
        Single.h)
