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

#include "./layer.hh"
#include "./network.hh"
#include "../functions.hh"

Network::Network(Weights* weights, int input_size, int output_size,
                int layer_size) {
    this->weights = weights;
    this->input_size = input_size;
    // Please mind the difference between output_size and layer_size
    // output_size is what you will really need after the propagation
    // e.g. if you want to have a word, it will be 26
    this->output_size = output_size;
    // layer_size is the number of neurons in the cell
    // not related to the output_size, but have to be superior to output_size
    this->layer_size = layer_size;
}

std::vector<Eigen::VectorXd> Network::propagate(std::vector<Eigen::VectorXd> inputs) {
    int size_net = inputs.size();
    // We initialize a null previous output
    Eigen::VectorXd previous_output = Eigen::VectorXd::Zero(this->layer_size);
    // Where outputs are stored
    std::vector<Eigen::VectorXd> outputs;
    // For each input
    for (int l = 0; l < size_net; l++) {
        // Creates as neuron layer
        this->layers.push_back(Layer(weights));
        // Propagating the information through the layer and saving the output
        previous_output = this->layers.back()
            .compute(inputs.at(l), previous_output);
        // Saving the output too (not optimal, should be rewritten after)
        outputs.push_back(previous_output);
    }
    return(outputs);
}

void Network::reset_layers() {
    // Destroy the layers, needed before the next propagation
    this->layers.clear();
}

void Network::backpropagate(std::vector<Eigen::VectorXd> expected_outputs) {
    // Do we have as many expected outputs as inputs ?
    if (expected_outputs.size() != layers.size()) {
        throw std::logic_error("Layer size != expected_outputs size");
    }
    // Initializing null gradient
    Eigen::VectorXd delta_prev = Eigen::VectorXd::Zero(layer_size);
    // For each cell, in descending order
    for (int l=expected_outputs.size()-1; l >= 0; l--) {
        // Computes the output cost function derivative
        Eigen::VectorXd delta = costfunction_derivative(
            expected_outputs.at(l), this->layers.at(l).output);
        // Adding the gradient computed with the one from the next layer
        delta = delta + delta_prev;
        // Backpropagating through the layer and retrieving the gradient
        delta_prev = this->layers.at(l).compute_gradient(delta);
    }
}
