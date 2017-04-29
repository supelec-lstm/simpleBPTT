// Copyright PinaPL
//
// weightsLSTM.cc
// PinaPL
//

#include <eigen3/Eigen/Dense>
#include <random>
#include "weightsLSTM.hh"

WeightsLSTM::WeightsLSTM(int input_size, int output_size) {
    this->input_size = input_size;
    this->output_size = output_size;

// We initialize random weights
    this->weight_in_forget_gate = 0.1 * Eigen::MatrixXd::Random(
        this->output_size,
        this->input_size);

    this->weight_in_input_gate = 0.1 * Eigen::MatrixXd::Random(
        this->output_size,
        this->input_size);

    this->weight_in_input_block = 0.1 * Eigen::MatrixXd::Random(
        this->output_size,
        this->input_size);

    this->weight_in_output_gate = 0.1 * Eigen::MatrixXd::Random(
        this->output_size,
        this->input_size);

    this->weight_st_forget_gate = 0.1 * Eigen::MatrixXd::Random(
        this->output_size,
        this->output_size);

    this->weight_st_input_gate = 0.1 * Eigen::MatrixXd::Random(
        this->output_size,
        this->output_size);

    this->weight_st_input_block = 0.1 * Eigen::MatrixXd::Random(
        this->output_size,
        this->output_size);

    this->weight_st_output_gate = 0.1 * Eigen::MatrixXd::Random(
        this->output_size,
        this->output_size);

/*
    this->bias_forget_gate = 0.1
        * Eigen::MatrixXd::Random(this->output_size, 1);
    this->bias_input_gate = 0.1
        * Eigen::MatrixXd::Random(this->output_size, 1);
    this->bias_input_block = 0.1
        * Eigen::MatrixXd::Random(this->output_size, 1);
    this->bias_output_gate = 0.1
        * Eigen::MatrixXd::Random(this->output_size, 1);
*/


// We initialize a null gradient

    this->delta_weight_in_forget_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size);

    this->delta_weight_in_input_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size);

    this->delta_weight_in_input_block = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size);

    this->delta_weight_in_output_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size);

    this->delta_weight_st_forget_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->output_size);

    this->delta_weight_st_input_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->output_size);

    this->delta_weight_st_input_block = Eigen::MatrixXd::Zero(
        this->output_size,
        this->output_size);

    this->delta_weight_st_output_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->output_size);
/*
    this->delta_bias_forget_gate = Eigen::MatrixXd::Zero(this->output_size, 1);
    this->delta_bias_input_gate = Eigen::MatrixXd::Zero(this->output_size, 1);
    this->delta_bias_input_block = Eigen::MatrixXd::Zero(this->output_size, 1);
    this->delta_bias_output_gate = Eigen::MatrixXd::Zero(this->output_size, 1);
*/
}
void WeightsLSTM::apply_gradient(double lambda) {
// We apply the weight variations
    this->weight_in_forget_gate =
        this->weight_in_forget_gate
        - lambda * this->delta_weight_in_forget_gate;

    this->weight_in_input_gate =
        this->weight_in_input_gate
        - lambda * this->delta_weight_in_input_gate;

    this->weight_in_input_block =
        this->weight_in_input_block
        - lambda * this->delta_weight_in_input_block;

    this->weight_in_output_gate =
        this->weight_in_output_gate
        - lambda * this->delta_weight_in_output_gate;

    this->weight_st_forget_gate =
        this->weight_st_forget_gate
        - lambda * this->delta_weight_st_forget_gate;

    this->weight_st_input_gate =
        this->weight_st_input_gate
        - lambda * this->delta_weight_st_input_gate;

    this->weight_st_input_block =
        this->weight_st_input_block
        - lambda * this->delta_weight_st_input_block;

    this->weight_st_output_gate =
        this->weight_st_output_gate
        - lambda * this->delta_weight_st_output_gate;

/*
    this->bias_forget_gate -= lambda * this->delta_bias_forget_gate;
    this->bias_input_gate -= lambda * this->delta_bias_input_gate;
    this->bias_input_block -= lambda * this->delta_bias_input_block;
    this->bias_output_gate -= lambda * this->delta_bias_output_gate;
*/

// We set a null gradient
    this->delta_weight_in_forget_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size);

    this->delta_weight_in_input_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size);

    this->delta_weight_in_input_block = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size);

    this->delta_weight_in_output_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size);

    this->delta_weight_st_forget_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->output_size);

    this->delta_weight_st_input_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->output_size);

    this->delta_weight_st_input_block = Eigen::MatrixXd::Zero(
        this->output_size,
        this->output_size);

    this->delta_weight_st_output_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->output_size);
/*
    this->bias_forget_gate = Eigen::MatrixXd::Zero(this->output_size, 1);
    this->bias_input_gate = Eigen::MatrixXd::Zero(this->output_size, 1);
    this->bias_input_block = Eigen::MatrixXd::Zero(this->output_size, 1);
    this->bias_output_gate = Eigen::MatrixXd::Zero(this->output_size, 1);
*/
}
