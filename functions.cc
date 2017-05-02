// Copyright PinaPL
//
// functions.cc
// PinaPL
//
#include <math.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <stdexcept>
#include <string>

double sigmoid(double x) {
    return (1/(1+exp(-x)));
}

double sigmoid_fast(double x) {
    return (0.5 * tanh(x) + 0.5);
}

double sigmoid_very_fast(double x) {
    return (0.5 * x/(1+std::abs(x)) + 0.5);
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
    // We have two cases :
    // The layer size is the output size
    if (expected_output.size() == output.size()) {
        // in this case, everything is ok, just return the cost
        return((expected_output-output).cwiseProduct(expected_output-output));
    // If not, we have to make a little hack
    } else {
        int layer_size = output.size();
        // We truncate the output
        // We compute the cost function
        // We come back to the original size
        return(Eigen::MatrixXd::Identity(layer_size, expected_output.size())
        * (((Eigen::MatrixXd::Identity(expected_output.size(), layer_size)
        * output)-expected_output).array().pow(2).matrix()));
    }
}

Eigen::VectorXd costfunction_derivative(Eigen::VectorXd expected_output,
                             Eigen::VectorXd output) {
    // We have two cases
    // The layer size is the output size
    if (expected_output.size() == output.size()) {
        // in this case, everything is ok, just return the cost derivative
        return((expected_output-output).cwiseProduct(expected_output-output));
    // If not, we have to make a little hack
    } else {
        int layer_size = output.size();
        // We truncate the output
        // We compute the costfunction derivative
        // We come back to the original size
        return(- Eigen::MatrixXd::Identity(layer_size, expected_output.size())
        * (((Eigen::MatrixXd::Identity(expected_output.size(), layer_size)
        * output)-expected_output)));
    }
}

double threshold(double x) {
    double level = 0.3;
    if (x > level) {
        return (1);
    } else {
        return(0);
    }
}

std::string open_file(bool dual) {
    if (dual) return("symmetrical_reber_train_2.4M.txt");
    else return("reber_train_2.4M.txt");
}
