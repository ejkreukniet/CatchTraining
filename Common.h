#pragma once

#include <string>

#include <Eigen/Dense>

using Eigen::MatrixXd;

#define assertValue(message, expected, value) \
    std::cout << message << ": " << value << " (" << expected << ")" << std::endl; \
    assert(abs((expected - value) / (expected + value)) < 1E-4);

MatrixXd load_csv(const std::string &path);
