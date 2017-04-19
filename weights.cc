// Copyright PinaPL
//
// weights.cc
// PinaPL
//

#include <eigen3/Eigen/Dense>
#include <random>
#include <stdexcept>
#include "weights.hh"

Weights::Weights(int input_size, int output_size) {
    this->input_size = input_size;
    this->output_size = output_size;

// We initialize random weights
    this->weight_input = 0.1 * Eigen::MatrixXd::Random(
        this->output_size,
        this->input_size);

    this->weight_prev = 0.1 * Eigen::MatrixXd::Random(
        this->output_size,
        this->output_size);

    this->bias = 0.1
        * Eigen::MatrixXd::Random(this->output_size, 1);

// We initialize a null gradient

    this->delta_weight_input = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size);

    this->delta_weight_prev = Eigen::MatrixXd::Zero(
        this->output_size,
        this->output_size);

    this->delta_bias = Eigen::MatrixXd::Zero(this->output_size, 1);
}

void Weights::apply_gradient(double lambda) {
// We apply the weight variations
    this->weight_input =
        this->weight_input
        + lambda * this->delta_weight_input;

    this->weight_prev =
        this->weight_prev
        + lambda * this->delta_weight_prev;

    this->bias = this->bias + lambda * this->delta_bias;
// We set a null gradient
    this->delta_weight_input = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size);

    this->delta_weight_prev = Eigen::MatrixXd::Zero(
        this->output_size,
        this->output_size);

    this->delta_bias = Eigen::MatrixXd::Zero(this->output_size, 1);
}
