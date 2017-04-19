// Copyright PinaPL
//
// functions.cc
// PinaPL
//
#include <math.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <stdexcept>

double sigmoid(double x) {
    return (1/(1+exp(-x)));
}

double sigmoid_derivative(double x) {
    return x*(1-x);
}

double tanh_derivative(double x) {
    return 1-tanh(x)*tanh(x);
}

double tanh2(double x) {
    double y = tanh(x);
    return y*y;
}

double tanhyp(double x) {
    return tanh(x);
}

Eigen::VectorXd costfunction(Eigen::VectorXd expected_output,
                             Eigen::VectorXd output) {
    if (expected_output.size() == output.size()) {
        return((expected_output-output).cwiseProduct(expected_output-output));
    } else {
        int layer_size = output.size();
        return(Eigen::MatrixXd::Identity(layer_size, expected_output.size())
        * (((Eigen::MatrixXd::Identity(expected_output.size(), layer_size)
        * output)-expected_output).array().pow(2).matrix()));
    }
}

Eigen::VectorXd costfunction_derivative(Eigen::VectorXd expected_output,
                             Eigen::VectorXd output) {
    if (expected_output.size() == output.size()) {
        return((expected_output-output).cwiseProduct(expected_output-output));
    } else {
        int layer_size = output.size();
        return(- Eigen::MatrixXd::Identity(layer_size, expected_output.size())
        * (((Eigen::MatrixXd::Identity(expected_output.size(), layer_size)
        * output)-expected_output)));
    }
}
