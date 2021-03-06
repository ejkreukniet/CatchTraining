
#include <iostream>
#include <string>

#include <Eigen/Dense>

#include "Common.h"

#include "catch.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

MatrixXd sigmoid(MatrixXd z)
{
    return (1 + (-z).array().exp()).inverse().matrix();
}

double computeCostSigmoid(MatrixXd X, MatrixXd y, VectorXd theta, VectorXd &grad)
{
    int m = (int) X.rows(); // Number of training samples

    MatrixXd h = sigmoid(X * theta);

    VectorXd J = (-y.transpose() * h.array().log().matrix()
        - (1 - y.array()).matrix().transpose() * (1 - h.array()).log().matrix()) / m;

    grad = (X.transpose() * (h - y)) / m;

    return J(0);
}

double gradientDescentLogistic(MatrixXd X, MatrixXd y, VectorXd &theta, double alpha, int iterations)
{
    int m = (int) X.rows(); // Number of training samples

    VectorXd J_history = VectorXd::Zero(iterations);

    VectorXd grad;

    for (int i = 0; i < iterations; ++i) {
        theta -= X.transpose() * (sigmoid(X * theta) - y) * (alpha / m);

        J_history(i) = computeCostSigmoid(X, y, theta, grad);
    }

//    std::cout << "J_history: " << J_history << std::endl;

    return J_history(iterations - 1);
}

MatrixXd predictLabels(MatrixXd X, VectorXd theta)
{
    int m = (int) X.rows(); // Number of training samples

    MatrixXd h = sigmoid(X * theta);

    for (int i = 0; i < m; ++i) {
        h(i) = h(i) >= 0.5 ? 1 : 0;
    }

    return h;
}

TEST_CASE( "Logistic regression", "[LogisticRegression]" )
{
    const int NUMBER_OF_VALUES = 1;

    MatrixXd data = load_csv("./logistic.csv");
    int m = (int) data.rows(); // Number of training samples
    int n = (int) data.cols() - NUMBER_OF_VALUES; // Number of features

    REQUIRE(m == 100);
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
    VectorXd grad;

    double J = computeCostSigmoid(X, y, theta, grad);

    REQUIRE_THAT(J, Catch::Matchers::WithinAbs(0.6931, HIGH_ACCURACY));

    REQUIRE_THAT(grad(0), Catch::Matchers::WithinAbs(-0.1000, HIGH_ACCURACY));
    REQUIRE_THAT(grad(1), Catch::Matchers::WithinAbs(-12.0092, HIGH_ACCURACY));
    REQUIRE_THAT(grad(2), Catch::Matchers::WithinAbs(-11.2628, HIGH_ACCURACY));

    VectorXd test(n + 1);
    test << -24, 0.2, 0.2;

    J = computeCostSigmoid(X, y, test, grad);

    REQUIRE_THAT(J, Catch::Matchers::WithinAbs(0.2183, HIGH_ACCURACY));

    REQUIRE_THAT(grad(0), Catch::Matchers::WithinAbs(0.043, LOW_ACCURACY));
    REQUIRE_THAT(grad(1), Catch::Matchers::WithinAbs(2.566, LOW_ACCURACY));
    REQUIRE_THAT(grad(2), Catch::Matchers::WithinAbs(2.647, LOW_ACCURACY));

    // Feature normalize
    VectorXd mu = X.rightCols(n).colwise().mean();

    VectorXd sigma = ((X.rightCols(n).rowwise() - mu.transpose()).array().square().colwise().sum() / (m - 1)).sqrt();

    MatrixXd temp = X.rightCols(n).rowwise() - mu.transpose();

    for (int i = 0; i < m; ++i) {
        temp(i, 0) /= sigma(0);
        temp(i, 1) /= sigma(1);
    }

    X.block(0, 1, m, n) = temp;

    J = gradientDescentLogistic(X, y, theta, 0.01, 50000);

    REQUIRE_THAT(J, Catch::Matchers::WithinAbs(0.203, LOW_ACCURACY));

    REQUIRE_THAT(theta(0), Catch::Matchers::WithinAbs(1.68459, LOW_ACCURACY));
    REQUIRE_THAT(theta(1), Catch::Matchers::WithinAbs(3.94023, LOW_ACCURACY));
    REQUIRE_THAT(theta(2), Catch::Matchers::WithinAbs(3.67359, LOW_ACCURACY));

    VectorXd predict(n + 1);
    predict << 1, 45, 85; // Predict admission for a student with scores 45 and 85
    predict(1) = (predict(1) - mu(0)) / sigma(0);
    predict(2) = (predict(2) - mu(1)) / sigma(1);

    VectorXd prob = sigmoid(predict.transpose() * theta);

    REQUIRE_THAT(prob(0), Catch::Matchers::WithinAbs(0.775, 0.01));

    MatrixXd p = predictLabels(X, theta);

    // Train Accuracy
    REQUIRE(p.cwiseEqual(y).count() == 89);
}
