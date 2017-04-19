// Copyright PinaPL
//
// layer.cc
// PinaPL
//
#include <math.h>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <stdexcept>
#include "weights.hh"
#include "layer.hh"
#include "functions.hh"
// #include "functions.hh"

Layer::Layer(Weights* weights) {
    this->weights = weights;
}

Eigen::VectorXd Layer::compute(Eigen::VectorXd input, Eigen::VectorXd previous_output) {
    this->input = input;
    this->previous_output = previous_output;
    this->output = (this->weights->weight_input * this->input
                    + this->weights->weight_prev * this->previous_output + this->weights->bias).unaryExpr(&sigmoid);
    return(this->output);
}

Eigen::VectorXd Layer::compute_gradient(Eigen::VectorXd deltas) {
    Eigen::VectorXd delta_sigmoid = deltas.cwiseProduct(this->output)
                .cwiseProduct(Eigen::MatrixXd::Ones(this->weights->output_size, 1)- this->output);

    this->weights->delta_weight_input = this->weights->delta_weight_input + delta_sigmoid * this->input.transpose();
    this->weights->delta_weight_prev = this->weights->delta_weight_prev + delta_sigmoid * this->previous_output.transpose();
    this->weights->delta_bias = this->weights->delta_bias + delta_sigmoid;

    Eigen::VectorXd delta_input = this->weights->weight_prev.transpose() * delta_sigmoid;
    return(delta_input);
}
