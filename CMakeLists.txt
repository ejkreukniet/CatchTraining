cmake_minimum_required(VERSION 3.13)
project(MachineLearning)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(MachineLearning
        Common.cpp
        Common.h
        Logistic.cpp
        Logistic.h
        MachineLearning.cpp
        Multi.cpp
        Multi.h
        Single.cpp
        Single.h)
