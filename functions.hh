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
#include <string>

double sigmoid(double x);
double sigmoid_fast(double x);
double sigmoid_very_fast(double x);
double sigmoid_derivative(double x);
double tanh_derivative(double x);
double tanh2(double x);
double tanhyp(double x);
Eigen::VectorXd costfunction(Eigen::VectorXd expected_output, Eigen::VectorXd output);
Eigen::VectorXd costfunction_derivative(Eigen::VectorXd expected_output,
                             Eigen::VectorXd output);
double threshold(double x);
std::string open_file(bool dual);
#endif
