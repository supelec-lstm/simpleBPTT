// Copyright PinaPL
//
// cell.cc
// PinaPL
//
#include <math.h>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <stdexcept>
#include "./weightsLSTM.hh"
#include "./cell.hh"
#include "../functions.hh"

Cell::Cell(WeightsLSTM* weights) {
    this->weights = weights;
}

std::vector<Eigen::VectorXd> Cell::compute(Eigen::VectorXd previous_output,
                                           Eigen::VectorXd previous_cell_state,
                                           Eigen::VectorXd input) {
    // We keep those values for later
    this->input = input;
    this->previous_output = previous_output;

    // Computes the input gate output
    this->input_gate_out =
        ((this->weights->W_in_input_gate * input
        + this->weights->W_prev_input_gate * previous_cell_state)
        + this->weights->bias_input_gate).unaryExpr(&sigmoid);

    // Computes the input bloc output
    this->input_block_out =
        ((this->weights->W_in_input_block * input
        + this->weights->W_prev_input_block * previous_cell_state)
        + this->weights->bias_input_block).unaryExpr(&tanh);

    // Computes the output gate output
    this->output_gate_out =
        ((this->weights->W_in_output_gate * input
        + this->weights->W_prev_output_gate * previous_cell_state)
        + this->weights->bias_output_gate).unaryExpr(&sigmoid);

    // Computes the new cell state
    this->cell_state =
        (previous_cell_state/*.cwiseProduct(this->forget_gate_out)*/
        + this->input_gate_out.cwiseProduct(this->input_block_out));

    // Computes the cell output
    this->cell_out =
        this->cell_state.unaryExpr(&tanh).cwiseProduct(this->output_gate_out);

    // Craft the result to return [cell_out, cell_state]
    std::vector<Eigen::VectorXd> result;
    result.push_back(cell_out);
    result.push_back(cell_state);
    return result;
}

std::vector<Eigen::VectorXd> Cell::compute_gradient(Eigen::VectorXd deltas,
    Eigen::VectorXd previous_delta_cell_in,
    Eigen::VectorXd previous_delta_cell_state) {

    // Computes dy
    Eigen::VectorXd delta_cell_out = previous_delta_cell_in + deltas;

    // Computes do
    Eigen::VectorXd delta_output_gate =
        delta_cell_out.cwiseProduct(this->cell_state.unaryExpr(&tanh))
            .cwiseProduct(this->output_gate_out.unaryExpr(&sigmoid_derivative));

    // Computes and stores the weights' variations
    this->weights->delta_W_in_output_gate +=
        delta_output_gate * this->input.transpose();

    this->weights->delta_W_prev_output_gate +=
        delta_output_gate * this->previous_output.transpose();

    this->weights->delta_bias_output_gate = delta_output_gate
        + this->weights->delta_bias_output_gate;

    // Computes dc
    Eigen::VectorXd delta_cell_state = previous_delta_cell_state
        + delta_cell_out.cwiseProduct(this->output_gate_out)
        .cwiseProduct(this->cell_state.unaryExpr(&tanh_derivative));  // CHECKED

    // Computes df
    //    Eigen::VectorXd delta_forget_gate;

    // Computes di
    Eigen::VectorXd delta_input_gate = delta_cell_state.cwiseProduct(this->input_block_out)
        .cwiseProduct(this->input_gate_out.unaryExpr(&sigmoid_derivative));

    this->weights->delta_W_in_input_gate +=
        delta_input_gate * this->input.transpose();

    this->weights->delta_W_prev_input_gate +=
        delta_input_gate * this->previous_output.transpose();

    this->weights->delta_bias_input_gate = delta_input_gate
        + this->weights->delta_bias_input_gate;

    // Computes dz
    Eigen::VectorXd delta_input_block =
        delta_cell_state.cwiseProduct(this->input_gate_out).cwiseProduct(
            this->input_block_out.unaryExpr(&tanh_computed_derivative));

    this->weights->delta_W_in_input_block +=
        delta_input_block * this->input.transpose();

    this->weights->delta_W_prev_input_block +=
        delta_input_block * this->previous_output.transpose();

    this->weights->delta_bias_input_block = delta_input_block
        + this->weights->delta_bias_input_block;

    // Computes the previous output gradient
    Eigen::VectorXd delta_previous_output =
        this->weights->W_prev_input_block.transpose() * delta_input_block +
        this->weights->W_prev_input_gate.transpose() * delta_input_gate +
        // this->weights->weight_in_forget_gate * delta_forget_gate +
        this->weights->W_prev_output_gate.transpose() * delta_output_gate;


    // Crafting the result [delta_input, delta_cell_state]
    std::vector<Eigen::VectorXd> result;
    result.push_back(delta_previous_output);
    result.push_back(delta_cell_state);
    return result;
}
