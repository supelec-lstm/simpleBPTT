// Copyright PinaPL
//
// weights.hh
// PinaPL
//

#ifndef WEIGHTSLSTM_HH
#define WEIGHTSLSTM_HH

#include <stdlib.h>
#include <eigen3/Eigen/Dense>
#include <string>
#include <sstream>
#include <iostream>


class WeightsLSTM {
 public:
    WeightsLSTM(int input_size, int output_size);
    ~WeightsLSTM();
    void apply_gradient(double lambda);

    // NOTE : Ugly part, should be converted to getters

    //   Information :
    // W_in means the weight matrix applied to the new input
    // W_st means the weight matrix applied to the previous cell state
    Eigen::MatrixXd W_in_forget_gate;
    Eigen::MatrixXd W_in_input_gate;
    Eigen::MatrixXd W_in_input_block;
    Eigen::MatrixXd W_in_output_gate;

    Eigen::MatrixXd W_prev_forget_gate;
    Eigen::MatrixXd W_prev_input_gate;
    Eigen::MatrixXd W_prev_input_block;
    Eigen::MatrixXd W_prev_output_gate;

    //   Information :
    // W_in means the weight matrix applied to the new INPUT
    // W_st means the weight matrix applied to the previous cell STATE
    Eigen::MatrixXd delta_W_in_forget_gate;
    Eigen::MatrixXd delta_W_in_input_gate;
    Eigen::MatrixXd delta_W_in_input_block;
    Eigen::MatrixXd delta_W_in_output_gate;

    Eigen::MatrixXd delta_W_prev_forget_gate;
    Eigen::MatrixXd delta_W_prev_input_gate;
    Eigen::MatrixXd delta_W_prev_input_block;
    Eigen::MatrixXd delta_W_prev_output_gate;

    int input_size;
    int output_size;
};
#endif
