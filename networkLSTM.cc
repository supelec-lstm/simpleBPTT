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
    this->output_size = output_size;
    this->layer_size = layer_size;
}

std::vector<Eigen::VectorXd> NetworkLSTM::propagate(std::vector<Eigen::VectorXd> inputs) {
    int size_net = inputs.size();
    Eigen::VectorXd previous_output = Eigen::VectorXd::Zero(this->layer_size);
    Eigen::VectorXd previous_cell_state = Eigen::VectorXd::Zero(this->layer_size);
    std::vector<Eigen::VectorXd> outputs;
    std::vector<Eigen::VectorXd> propagation_result;
    for (int l = 0; l < size_net; l++) {
        this->cells.push_back(Cell(weights));
        propagation_result = this->cells.back().compute(previous_output, previous_cell_state, inputs.at(l));
        previous_output = propagation_result.at(0);
        previous_cell_state = propagation_result.at(1);
        outputs.push_back(previous_output);
    }
    return(outputs);
}

void NetworkLSTM::reset_layers() {
    this->cells.clear();
}
/* TODO(Hugo Shaka): adapt backprop to LSTM
void Network::backpropagate(std::vector<Eigen::VectorXd> expected_outputs) {
    if (expected_outputs.size() != layers.size()) {
        throw std::logic_error("Layer size != expected_outputs size");
    }
    Eigen::VectorXd delta_prev = Eigen::VectorXd::Zero(layer_size);
    for (int l=expected_outputs.size()-1; l >= 0; l--) {
        Eigen::VectorXd delta = costfunction_derivative(
            expected_outputs.at(l), this->layers.at(l).output);
        delta = delta + delta_prev;
        delta_prev = this->layers.at(l).compute_gradient(delta);
        // this->layers.at(l).compute_weight_gradient();
    }
}
*/
