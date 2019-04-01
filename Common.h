#pragma once

#include <string>

#include <Eigen/Dense>

using Eigen::MatrixXd;

MatrixXd load_csv(const std::string &path);
