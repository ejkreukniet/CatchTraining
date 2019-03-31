
#include <iostream>
#include <string>

#include <Eigen/Dense>

#include "Common.h"

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

void testLogisticRegression()
{
    const int NUMBER_OF_VALUES = 1;

    MatrixXd data = load_csv("../logistic.csv");
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

    double J = computeCostSigmoid(X, y, theta);
    assertValue("Cost 1", 0.6931, J);

    VectorXd test(n + 1);
    test << -24, 0.2, 0.2;

    J = computeCostSigmoid(X, y, test);
    assertValue("Cost 2", 0.2183, J);

    // Feature normalize
    VectorXd mu = X.rightCols(n).colwise().mean();
    std::cout << "mu: " << mu << std::endl;

    VectorXd sigma = ((X.rightCols(n).rowwise() - mu.transpose()).array().square().colwise().sum() / (m - 1)).sqrt();
    std::cout << "sigma: " << sigma << std::endl;

    MatrixXd temp = X.rightCols(n).rowwise() - mu.transpose();

    for (int i = 0; i < m; ++i) {
        temp(i, 0) /= sigma(0);
        temp(i, 1) /= sigma(1);
    }

    theta = gradientDescentLogistic(X, y, theta, 0.01, 400);
    std::cout << "theta: " << theta << std::endl;
    std::cout << "cost: " << computeCostSigmoid(X, y, theta) << std::endl; // 0.203

    VectorXd predict(n + 1);
    predict << 1, 45, 85;

    VectorXd prob = sigmoid(predict.transpose() * theta);
    std::cout << "prob: " << prob << std::endl;

    MatrixXd p = predictLabels(X, theta);

    int c = 0;
    for (int i = 0; i < m; ++i) {
        if (p(i) == y(i)) ++c;
    }
    std::cout << "Correct: " << (double) c << std::endl << (double) m << std::endl;
}
