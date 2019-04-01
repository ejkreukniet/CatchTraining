
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

double computeCostSigmoid(MatrixXd X, MatrixXd y, VectorXd theta)
{
    int m = (int) X.rows(); // Number of training samples

    MatrixXd h = sigmoid(X * theta);

    // J = (-y' *log(h) - (1-y)' * log(1-h))/m;
    // grad = (X'*(h - y))/m;
    VectorXd J = (-y.transpose() * h.array().log().matrix()
        - (1 - y.array()).matrix().transpose() * (1 - h.array()).log().matrix()) / m;

    return J(0);
}

VectorXd gradientDescentLogistic(MatrixXd X, MatrixXd y, VectorXd theta, double alpha, int iterations)
{
    int m = (int) X.rows(); // Number of training samples

    VectorXd J_history = VectorXd::Zero(iterations);

    for (int i = 0; i < iterations; ++i) {
        theta -= X.transpose() * (sigmoid(X * theta) - y) * (alpha / m);

        J_history(i) = computeCostSigmoid(X, y, theta);
    }
//std::cout << "J_history: " << J_history << std::endl;
    return theta;
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

TEST_CASE( "Logistic Regression", "[LogisticRegression]" )
{
    const int NUMBER_OF_VALUES = 1;

    MatrixXd data = load_csv("../logistic.csv");
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

    //std::cout << "X: " << X.block(0, 0, 10, n + 1) << std::endl;
    //std::cout << "y: " << y.block(0, 0, 10, NUMBER_OF_VALUES) << std::endl;
    //std::cout << "theta: " << theta << std::endl;

    double J = computeCostSigmoid(X, y, theta);

    REQUIRE_THAT(J, Catch::Matchers::WithinAbs(0.6931, 0.0001));

    VectorXd test(n + 1);
    test << -24, 0.2, 0.2;

    J = computeCostSigmoid(X, y, test);

    REQUIRE_THAT(J, Catch::Matchers::WithinAbs(0.2183, 0.0001));

    // Feature normalize
    VectorXd mu = X.rightCols(n).colwise().mean();
    std::cout << "mu: " << mu << std::endl;
    //assertValue("mu 0", 2000.6809, mu(0));
    //assertValue("mu 1", 3.1702, mu(1));

    VectorXd sigma = ((X.rightCols(n).rowwise() - mu.transpose()).array().square().colwise().sum() / (m - 1)).sqrt();
    std::cout << "sigma: " << sigma << std::endl;
    //assertValue("sigma 0", 794.70235, sigma(0));
    //assertValue("sigma 1", 0.76098, sigma(1));
//mu: 65.6443
// 66.222
//sigma: 19.4582
//18.5828

    MatrixXd temp = X.rightCols(n).rowwise() - mu.transpose();

    for (int i = 0; i < m; ++i) {
        temp(i, 0) /= sigma(0);
        temp(i, 1) /= sigma(1);
    }

    //X.block(0, 1, m, n) = temp;

    theta = gradientDescentLogistic(X, y, theta, 0.01, 400);
    std::cout << "theta: " << theta << std::endl;
    std::cout << "cost: " << computeCostSigmoid(X, y, theta) << std::endl; // 0.203


//	assertValue("Theta 0", -25.161, theta(0));
//	assertValue("Theta 1", 0.206, theta(1));
//	assertValue("Theta 2", 0.201, theta(2));

    VectorXd predict(n + 1);
    predict << 1, 45, 85;
    //predict(1) = (predict(1) - mu(0)) / sigma(0);
    //predict(2) = (predict(2) - mu(1)) / sigma(1);

    VectorXd prob = sigmoid(predict.transpose() * theta);
    std::cout << "prob: " << prob << std::endl;
    //assertValue("Predict admission for a student with scores 45 and 85", 0.775, prob(0));

    MatrixXd p = predictLabels(X, theta);
//std::cout << "p == y: " << (p.cwiseEqual(y)) << std::endl;
    //std::cout << "y: " << y << std::endl;
    //std::cout << "p: " << p << std::endl;
    int c = 0;
    for (int i = 0; i < m; ++i) {
        if (p(i) == y(i)) ++c;
    }
    std::cout << "Correct: " << (double) c << std::endl << (double) m << std::endl;

    //assertValue('Training accuracy:', 89.0, p.cwiseEqual(y).mean() * 100);
}
