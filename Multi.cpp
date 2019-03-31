
#include <iostream>
#include <string>

#include <Eigen/Dense>

#include "Common.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

double computeCostMulti(MatrixXd X, MatrixXd y, VectorXd theta)
{
    int m = (int) X.rows(); // Number of training samples

    MatrixXd d = X * theta - y;

    VectorXd J = d.transpose() * d / (2 * m);

    return J(0);
}

VectorXd gradientDescentMulti(MatrixXd X, MatrixXd y, VectorXd theta, double alpha, int iterations)
{
    int m = (int) X.rows(); // Number of training samples

    VectorXd J_history = VectorXd::Zero(iterations);

    for (int i = 0; i < iterations; ++i) {
        theta -= X.transpose() * (X * theta - y) * (alpha / m);

        J_history(i) = computeCostMulti(X, y, theta);
    }

    return theta;
}

VectorXd normalEquation(MatrixXd X, MatrixXd y)
{
    return (X.transpose() * X).inverse() * X.transpose() * y;
}

void testLinearRegressionWithMultipleFeatures()
{
    const int NUMBER_OF_VALUES = 1;

    MatrixXd data = load_csv("../multiple-features.csv");
    int m = (int) data.rows(); // Number of training samples
    int n = (int) data.cols() - NUMBER_OF_VALUES; // Number of features

    std::cout << "\nTraining samples: " << m << ", Features: " << n << std::endl;

    // Features
    MatrixXd X(m, n + 1);
    // Add intercept term to X
    X.block(0, 0, m, 1) = MatrixXd::Ones(m, 1);
    X.block(0, 1, m, n) = data.block(0, 0, m, n);

    // Values
    MatrixXd y = data.block(0, n, m, NUMBER_OF_VALUES);

    VectorXd theta(n + 1);
    theta << VectorXd::Zero(n + 1);

    // Feature normalize
    VectorXd mu = X.rightCols(n).colwise().mean();
    assertValue("mu 0", 2000.6809, mu(0));
    assertValue("mu 1", 3.1702, mu(1));

    VectorXd sigma = ((X.rightCols(n).rowwise() - mu.transpose()).array().square().colwise().sum() / (m - 1)).sqrt();
    assertValue("sigma 0", 794.70235, sigma(0));
    assertValue("sigma 1", 0.76098, sigma(1));

    MatrixXd temp = X.rightCols(n).rowwise() - mu.transpose();

    for (int i = 0; i < m; ++i) {
        temp(i, 0) /= sigma(0);
        temp(i, 1) /= sigma(1);
    }

    X.block(0, 1, m, n) = temp;

    std::cout << "\nWith gradient descent" << std::endl;

    theta = gradientDescentMulti(X, y, theta, 0.01, 400);
    assertValue("Theta 0", 334302.0, theta(0));
    assertValue("Theta 1", 100087.0, theta(1));
    assertValue("Theta 2", 3673.55, theta(2));

    VectorXd predict(n + 1);
    predict << 1, 1650, 3;
    predict(1) = (predict(1) - mu(0)) / sigma(0);
    predict(2) = (predict(2) - mu(1)) / sigma(1);

    double price = theta.transpose().dot(predict);
    assertValue("Predicted price of a 1650 sq-ft, 3 br house", 289314.620338, price);

    std::cout << "\nWith normal equation" << std::endl;

    // Calculate the parameters from the normal equation
    X.block(0, 0, m, 1) = MatrixXd::Ones(m, 1);
    X.block(0, 1, m, n) = data.block(0, 0, m, n);

    theta = normalEquation(X, y);
    assertValue("Theta 0", 89597.9, theta(0));
    assertValue("Theta 1", 139.211, theta(1));
    assertValue("Theta 2", -8738.02, theta(2));

    predict << 1, 1650, 3;

    price = theta.transpose().dot(predict);
    assertValue("Predicted price of a 1650 sq-ft, 3 br house (using normal equations)", 293081.464335, price);
}
