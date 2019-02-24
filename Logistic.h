#pragma once

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

MatrixXd sigmoid(MatrixXd z);
double computeCostSigmoid(MatrixXd X, MatrixXd y, VectorXd theta);
VectorXd gradientDescentLogistic(MatrixXd X, MatrixXd y, VectorXd theta, double alpha, int num_iters);

void testLogisticRegression();
