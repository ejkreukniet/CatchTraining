#pragma once

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

double computeCost(MatrixXd X, MatrixXd y, VectorXd theta);
VectorXd gradientDescent(MatrixXd X, MatrixXd y, VectorXd theta, double alpha, int num_iters);

void testLinearRegression();
