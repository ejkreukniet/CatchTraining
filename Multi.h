#pragma once

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

double computeCostMulti(MatrixXd X, MatrixXd y, VectorXd theta);
VectorXd gradientDescentMulti(MatrixXd X, MatrixXd y, VectorXd theta, double alpha, int num_iters);
VectorXd normalEquation(MatrixXd X, MatrixXd y);

void testLinearRegressionWithMultipleFeatures();
