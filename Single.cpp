
#include <iostream>
#include <string>

#include <Eigen/Dense>

#include "Common.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

double computeCost(MatrixXd X, MatrixXd y, VectorXd theta)
{
	int m = (int)X.rows(); // Number of training samples

	double J = (X * theta - y).array().pow(2).sum() / (2 * m);

	return J;
}

VectorXd gradientDescent(MatrixXd X, MatrixXd y, VectorXd theta, double alpha, int iterations)
{
	int m = (int)X.rows(); // Number of training samples

	VectorXd J_history = VectorXd::Zero(iterations);

	for (int i = 0; i < iterations; ++i)
	{
		theta -= (X.transpose() * (X * theta - y)) * (alpha / m);

		J_history(i) = computeCost(X, y, theta);
	}

	return theta;
}

void testLinearRegression()
{
	const int NUMBER_OF_VALUES = 1;

	MatrixXd data = load_csv("single-feature.csv");
	int m = (int)data.rows(); // Number of training samples
	int n = (int)data.cols() - NUMBER_OF_VALUES; // Number of features

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

	//std::cout << "X: " << X.block(0, 0, 10, n + 1) << std::endl;
	//std::cout << "y: " << y.block(0, 0, 10, NUMBER_OF_VALUES) << std::endl;
	//std::cout << "theta: " << theta << std::endl;

	double J = computeCost(X, y, theta);
	assertValue("Cost 1", 32.07, J);

	VectorXd test(n + 1);
	test << -1, 2;

	J = computeCost(X, y, test);
	assertValue("Cost 2", 54.24, J);

	theta = gradientDescent(X, y, theta, 0.01, 1500);
	assertValue("Theta 0", -3.6303, theta(0));
	assertValue("Theta 1", 1.1664, theta(1));

	MatrixXd predict1(1, n + 1);
	predict1 << 1, 3.5;
	assertValue("Predict profit for population 35,000", 4519.767868, (predict1 * theta * 10000.0)(0));

	MatrixXd predict2(1, n + 1);
	predict2 << 1, 7;
	assertValue("Predict profit for population 70,000", 45342.450129, (predict2 * theta * 10000.0)(0));
}
