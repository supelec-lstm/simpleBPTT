// Copyright PinaPL
//
// weightsLSTM.cc
// PinaPL
//

#include <eigen3/Eigen/Dense>
#include <random>
#include "./weightsLSTM.hh"

WeightsLSTM::WeightsLSTM(int input_size, int output_size) {
    this->input_size = input_size;
    this->output_size = output_size;

// We make sure the random seed is initialized
srand(time(NULL));
// We initialize random weights
    this->W_in_forget_gate = 0.1 * Eigen::MatrixXd::Random(
        this->output_size,
        this->input_size);

    this->W_in_input_gate = 0.1 * Eigen::MatrixXd::Random(
        this->output_size,
        this->input_size);

    this->W_in_input_block = 0.1 * Eigen::MatrixXd::Random(
        this->output_size,
        this->input_size);

    this->W_in_output_gate = 0.1 * Eigen::MatrixXd::Random(
        this->output_size,
        this->input_size);

    this->W_prev_forget_gate = 0.1 * Eigen::MatrixXd::Random(
        this->output_size,
        this->output_size);

    this->W_prev_input_gate = 0.1 * Eigen::MatrixXd::Random(
        this->output_size,
        this->output_size);

    this->W_prev_input_block = 0.1 * Eigen::MatrixXd::Random(
        this->output_size,
        this->output_size);

    this->W_prev_output_gate = 0.1 * Eigen::MatrixXd::Random(
        this->output_size,
        this->output_size);


    this->bias_forget_gate = 0.1
        * Eigen::VectorXd::Random(this->output_size);
    this->bias_input_gate = 0.1
        * Eigen::VectorXd::Random(this->output_size);
    this->bias_input_block = 0.1
        * Eigen::VectorXd::Random(this->output_size);
    this->bias_output_gate = 0.1
        * Eigen::VectorXd::Random(this->output_size);



// We initialize a null gradient

    this->delta_W_in_forget_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size);

    this->delta_W_in_input_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size);

    this->delta_W_in_input_block = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size);

    this->delta_W_in_output_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size);

    this->delta_W_prev_forget_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->output_size);

    this->delta_W_prev_input_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->output_size);

    this->delta_W_prev_input_block = Eigen::MatrixXd::Zero(
        this->output_size,
        this->output_size);

    this->delta_W_prev_output_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->output_size);

    this->delta_bias_forget_gate = Eigen::VectorXd::Zero(this->output_size);
    this->delta_bias_input_gate = Eigen::VectorXd::Zero(this->output_size);
    this->delta_bias_input_block = Eigen::VectorXd::Zero(this->output_size);
    this->delta_bias_output_gate = Eigen::VectorXd::Zero(this->output_size);
}
void WeightsLSTM::apply_gradient(double lambda) {
// We apply the weight variations
    this->W_in_forget_gate =
        this->W_in_forget_gate
        + lambda * this->delta_W_in_forget_gate;

    this->W_in_input_gate =
        this->W_in_input_gate
        + lambda * this->delta_W_in_input_gate;

    this->W_in_input_block =
        this->W_in_input_block
        + lambda * this->delta_W_in_input_block;

    this->W_in_output_gate =
        this->W_in_output_gate
        + lambda * this->delta_W_in_output_gate;

    this->W_prev_forget_gate =
        this->W_prev_forget_gate
        + lambda * this->delta_W_prev_forget_gate;

    this->W_prev_input_gate =
        this->W_prev_input_gate
        + lambda * this->delta_W_prev_input_gate;

    this->W_prev_input_block =
        this->W_prev_input_block
        + lambda * this->delta_W_prev_input_block;

    this->W_prev_output_gate =
        this->W_prev_output_gate
        + lambda * this->delta_W_prev_output_gate;


    this->bias_forget_gate -= lambda * this->delta_bias_forget_gate;
    this->bias_input_gate -= lambda * this->delta_bias_input_gate;
    this->bias_input_block -= lambda * this->delta_bias_input_block;
    this->bias_output_gate -= lambda * this->delta_bias_output_gate;


// We set a null gradient
    this->delta_W_in_forget_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size);

    this->delta_W_in_input_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size);

    this->delta_W_in_input_block = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size);

    this->delta_W_in_output_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size);

    this->delta_W_prev_forget_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->output_size);

    this->delta_W_prev_input_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->output_size);

    this->delta_W_prev_input_block = Eigen::MatrixXd::Zero(
        this->output_size,
        this->output_size);

    this->delta_W_prev_output_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->output_size);

    this->bias_forget_gate = Eigen::VectorXd::Zero(this->output_size);
    this->bias_input_gate = Eigen::VectorXd::Zero(this->output_size);
    this->bias_input_block = Eigen::VectorXd::Zero(this->output_size);
    this->bias_output_gate = Eigen::VectorXd::Zero(this->output_size);
}
