#include <iostream>
#include <Eigen/Dense>
#include <utility>
#include <Eigen/Core>
#include <vector>
#include <utility>

#ifndef TOY_ML_FRAMEWORK_CONV_H
#define TOY_ML_FRAMEWORK_CONV_H

#endif //TOY_ML_FRAMEWORK_CONV_H

using namespace std;
using namespace Eigen;

MatrixXd img2col2D(const MatrixXd &input, const vector<MatrixXd> &kernels,
                           const int &kernel_H, const int &kernel_W,
                           const int &paddle, const int &stride);

MatrixXd MaxPooling2D(const MatrixXd &input, const int filter_H,
                      const int filter_W, const int stride,
                      const int img_H, const int img_W);


