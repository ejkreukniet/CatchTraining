
#include <iostream>
#include <string>

#include <Eigen/Dense>

#include "Common.h"

#include "catch.hpp"

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

TEST_CASE("Multiple features", "[MultipleFeatures]")
{
    const int NUMBER_OF_VALUES = 1;

    MatrixXd data = load_csv("./multiple-features.csv");
    int m = (int) data.rows(); // Number of training samples
    int n = (int) data.cols() - NUMBER_OF_VALUES; // Number of features

    REQUIRE(m == 47);
    REQUIRE(n == 2);

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

    REQUIRE_THAT(mu(0), Catch::Matchers::WithinAbs(2000.6809, HIGH_ACCURACY));
    REQUIRE_THAT(mu(1), Catch::Matchers::WithinAbs(3.1702, HIGH_ACCURACY));

    VectorXd sigma = ((X.rightCols(n).rowwise() - mu.transpose()).array().square().colwise().sum() / (m - 1)).sqrt();

    REQUIRE_THAT(sigma(0), Catch::Matchers::WithinAbs(794.70235, HIGH_ACCURACY));
    REQUIRE_THAT(sigma(1), Catch::Matchers::WithinAbs(0.76098, HIGH_ACCURACY));

    MatrixXd temp = X.rightCols(n).rowwise() - mu.transpose();

    for (int i = 0; i < m; ++i) {
        temp(i, 0) /= sigma(0);
        temp(i, 1) /= sigma(1);
    }

    X.block(0, 1, m, n) = temp;

    SECTION("With gradient descent") {
        theta = gradientDescentMulti(X, y, theta, 0.01, 400);

        REQUIRE_THAT(theta(0), Catch::Matchers::WithinAbs(334302.0639, HIGH_ACCURACY));
        REQUIRE_THAT(theta(1), Catch::Matchers::WithinAbs(100087.116, HIGH_ACCURACY));
        REQUIRE_THAT(theta(2), Catch::Matchers::WithinAbs(3673.5484, HIGH_ACCURACY));

        VectorXd predict(n + 1);
        predict << 1, 1650, 3; // Predicted price of a 1650 sq-ft, 3 br house
        predict(1) = (predict(1) - mu(0)) / sigma(0);
        predict(2) = (predict(2) - mu(1)) / sigma(1);

        double price = theta.transpose().dot(predict);

        REQUIRE_THAT(price, Catch::Matchers::WithinAbs(289314.620338, HIGH_ACCURACY));
    }

    SECTION("With normal equation") {
        // Calculate the parameters from the normal equation
        X.block(0, 0, m, 1) = MatrixXd::Ones(m, 1);
        X.block(0, 1, m, n) = data.block(0, 0, m, n);

        theta = normalEquation(X, y);

        REQUIRE_THAT(theta(0), Catch::Matchers::WithinAbs(89597.9095, HIGH_ACCURACY));
        REQUIRE_THAT(theta(1), Catch::Matchers::WithinAbs(139.2107, HIGH_ACCURACY));
        REQUIRE_THAT(theta(2), Catch::Matchers::WithinAbs(-8738.0191, HIGH_ACCURACY));

        VectorXd predict(n + 1);
        predict << 1, 1650, 3; // Predicted price of a 1650 sq-ft, 3 br house

        double price = theta.transpose().dot(predict);

        REQUIRE_THAT(price, Catch::Matchers::WithinAbs(293081.464335, HIGH_ACCURACY));
    }
}
