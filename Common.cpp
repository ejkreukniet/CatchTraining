
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::Map;
using Eigen::Matrix;
using Eigen::RowMajor;

MatrixXd load_csv(const std::string &path)
{
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    int rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename MatrixXd::Scalar,
                            MatrixXd::RowsAtCompileTime,
                            MatrixXd::ColsAtCompileTime,
                            RowMajor>>(values.data(), rows, values.size() / rows);
}
