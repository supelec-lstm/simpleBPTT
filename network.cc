// Copyright PinaPL
//
// network.cc
// PinaPL
//

#include <math.h>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <iostream>
#include <stdexcept>

#include "layer.hh"
#include "network.hh"
#include "functions.hh"

Network::Network(Weights* weights, int input_size, int output_size,
                int layer_size) {
    this->weights = weights;
    this->input_size = input_size;
    this->output_size = output_size;
    this->layer_size = layer_size;
}

std::vector<Eigen::VectorXd> Network::propagate(std::vector<Eigen::VectorXd> inputs) {
    int size_net = inputs.size();
    Eigen::VectorXd previous_output = Eigen::VectorXd::Zero(this->layer_size);
    std::vector<Eigen::VectorXd> outputs;
    for (int l = 0; l < size_net; l++) {
        this->layers.push_back(Layer(weights));
        previous_output = this->layers.back()
            .compute(inputs.at(l), previous_output);
        outputs.push_back(previous_output);
    }
    return(outputs);
}

void Network::reset_layers() {
    this->layers.clear();
}

void Network::backpropagate(std::vector<Eigen::VectorXd> expected_outputs) {
    if (expected_outputs.size() != layers.size()) {
        throw std::logic_error("Layer size != expected_outputs size");
    }
    Eigen::VectorXd delta_prev = Eigen::VectorXd::Zero(layer_size);
    for (int l=expected_outputs.size()-1; l >= 0; l--) {
        Eigen::VectorXd delta = costfunction(
            expected_outputs.at(l), this->layers.at(l).output);
        delta = delta + delta_prev;
        delta_prev = this->layers.at(l).compute_gradient(delta);
        // this->layers.at(l).compute_weight_gradient();
    }
    this->weights->apply_gradient(0.1);
}
