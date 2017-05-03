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

#include "cell.hh"
#include "networkLSTM.hh"
#include "functions.hh"
#include "weightsLSTM.hh"

NetworkLSTM::NetworkLSTM(WeightsLSTM* weights, int input_size, int output_size,
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

std::vector<Eigen::VectorXd> NetworkLSTM::propagate(std::vector<Eigen::VectorXd> inputs) {
    int size_net = inputs.size();
    // We initialize null memory and previous output
    Eigen::VectorXd previous_output = Eigen::VectorXd::Zero(this->layer_size);
    Eigen::VectorXd previous_cell_state = Eigen::VectorXd::Zero(this->layer_size);
    // Where the outputs will be stored, will be returned in the end
    std::vector<Eigen::VectorXd> outputs;
    // Temporary vector, in order to store the propagation result before using it
    std::vector<Eigen::VectorXd> propagation_result;
    // For each input
    for (int l = 0; l < size_net; l++) {
        // Creates a new LSTM cell
        this->cells.push_back(Cell(weights));
        // Propagates
        propagation_result = this->cells.back()
            .compute(previous_output, previous_cell_state, inputs.at(l));
        // Storing the previous output
        previous_output = propagation_result.at(0);
        // Storing the previous cell state
        previous_cell_state = propagation_result.at(1);
        // We may need the outputs for each cell
        outputs.push_back(previous_output);
    }
    return(outputs);
}

void NetworkLSTM::reset_layers() {
    // Destroy the cells, needed before the next propagation
    this->cells.clear();
}
// TODO(Hugo Shaka): adapt backprop to LSTM
void NetworkLSTM::backpropagate(std::vector<Eigen::VectorXd> expected_outputs) {
    // Do we have as many expected outputs as inputs ?
    if (expected_outputs.size() != cells.size()) {
        throw std::logic_error("Layer size != expected_outputs size");
    }
    // Initializing null gradient
    std::vector<Eigen::VectorXd> result;
    Eigen::VectorXd previous_delta_cell_in = Eigen::VectorXd::Zero(layer_size);
    Eigen::VectorXd previous_delta_cell_state = Eigen::VectorXd::Zero(layer_size);
    // For each cell, in descending order
    for (int l=expected_outputs.size()-1; l >= 0; l--) {
        // Computes the output cost function derivative
        Eigen::VectorXd delta = costfunction_derivative(
            expected_outputs.at(l), this->cells.at(l).cell_out);
        // Backpropagating through the layer and retrieving the gradients
        result = this->cells.at(l).compute_gradient(delta,
            previous_delta_cell_in, previous_delta_cell_state);
        previous_delta_cell_in = result.at(0);
        previous_delta_cell_state = result.at(1);
    }
}
