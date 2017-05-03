// Copyright PinaPL
//
// layer.cc
// PinaPL
//
#include <math.h>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <stdexcept>
#include "./weights.hh"
#include "./layer.hh"
#include "../functions.hh"
// #include "functions.hh"

Layer::Layer(Weights* weights) {
    this->weights = weights;
}

Eigen::VectorXd Layer::compute(Eigen::VectorXd input, Eigen::VectorXd previous_output) {
    this->input = input;
    this->previous_output = previous_output;
    // Product of (inputs concatenated with previous_output) and Weights
    this->output = (this->weights->weight_input * this->input
        + this->weights->weight_prev * this->previous_output + this->weights->bias)
        // Then applying activation function (sigmoid)
        .unaryExpr(&sigmoid);
    return(this->output);
}

Eigen::VectorXd Layer::compute_gradient(Eigen::VectorXd deltas) {
    // Computes the gradient after the sigmoid
    Eigen::VectorXd delta_sigmoid = deltas.cwiseProduct(this->output)
                .cwiseProduct(Eigen::MatrixXd::Ones(this->weights->output_size, 1)- this->output);

    // Computing the delta weights, they are stored in weight (not in the layer)
    this->weights->delta_weight_input =
        this->weights->delta_weight_input + delta_sigmoid * this->input.transpose();
    this->weights->delta_weight_prev =
        this->weights->delta_weight_prev + delta_sigmoid * this->previous_output.transpose();
    this->weights->delta_bias = this->weights->delta_bias + delta_sigmoid;

    // Computing the intput gradient
    Eigen::VectorXd delta_input = this->weights->weight_prev.transpose() * delta_sigmoid;
    return(delta_input);
}
