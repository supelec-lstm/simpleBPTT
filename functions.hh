// Copyright PinaPL
//
// functions.hpp
// PinaPL
//

#ifndef FUNCTIONS_HH
#define FUNCTIONS_HH

#include <math.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <stdexcept>

double sigmoid(double x);
double sigmoid_derivative(double x);
double tanh_derivative(double x);
double tanh2(double x);
double tanhyp(double x);
Eigen::VectorXd costfunction(Eigen::VectorXd expected_output, Eigen::VectorXd output);

#endif
