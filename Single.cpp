
#include <iostream>
#include <string>

#include <Eigen/Dense>

#include "Common.h"

#include "catch.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

double computeCost(MatrixXd X, MatrixXd y, VectorXd theta)
{
    int m = (int) X.rows(); // Number of training samples

    double J = (X * theta - y).array().pow(2).sum() / (2 * m);

    return J;
}

VectorXd gradientDescent(MatrixXd X, MatrixXd y, VectorXd theta, double alpha, int iterations)
{
    int m = (int) X.rows(); // Number of training samples

    VectorXd J_history = VectorXd::Zero(iterations);

    for (int i = 0; i < iterations; ++i) {
        theta -= (X.transpose() * (X * theta - y)) * (alpha / m);

        J_history(i) = computeCost(X, y, theta);
    }

    return theta;
}

TEST_CASE( "Single feature", "[SingleFeature]" )
{
    const int NUMBER_OF_VALUES = 1;

    MatrixXd data = load_csv("./single-feature.csv");
    int m = (int) data.rows(); // Number of training samples
    int n = (int) data.cols() - NUMBER_OF_VALUES; // Number of features

    REQUIRE(m == 97);
    REQUIRE(n == 1);

    // Features
    MatrixXd X(m, n + 1);
    // Add intercept term to X
    X.block(0, 0, m, 1) = MatrixXd::Ones(m, 1);
    X.block(0, 1, m, n) = data.block(0, 0, m, n);

    // Values
    MatrixXd y = data.block(0, n, m, NUMBER_OF_VALUES);

    VectorXd theta(n + 1);
    theta << VectorXd::Zero(n + 1);

    double J = computeCost(X, y, theta);

    REQUIRE_THAT(J, Catch::Matchers::WithinAbs(32.0727, HIGH_ACCURACY));

    VectorXd test(n + 1);
    test << -1, 2;

    J = computeCost(X, y, test);

    REQUIRE_THAT(J, Catch::Matchers::WithinAbs(54.2425, HIGH_ACCURACY));

    theta = gradientDescent(X, y, theta, 0.01, 1500);

    REQUIRE_THAT(theta(0), Catch::Matchers::WithinAbs(-3.6303, HIGH_ACCURACY));
    REQUIRE_THAT(theta(1), Catch::Matchers::WithinAbs(1.1664, HIGH_ACCURACY));

    MatrixXd predict1(1, n + 1);
    predict1 << 1, 3.5; // Predict profit for population 35,000

    REQUIRE_THAT((predict1 * theta * 10000.0)(0), Catch::Matchers::WithinAbs(4519.767868, HIGH_ACCURACY));

    MatrixXd predict2(1, n + 1);
    predict2 << 1, 7; // Predict profit for population 70,000

    REQUIRE_THAT((predict2 * theta * 10000.0)(0), Catch::Matchers::WithinAbs(45342.450129, HIGH_ACCURACY));
}
