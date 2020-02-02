#pragma once

#include <string>

#include "Eigen/Dense"

using Eigen::MatrixXd;

#define HIGH_ACCURACY 0.0001
#define LOW_ACCURACY 0.001

MatrixXd load_csv(const std::string &path);
