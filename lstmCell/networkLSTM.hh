// Copyright PinaPL
//
// layer.hh
// PinaPL
//
#ifndef NETWORKLSTM_HH
#define NETWORKLSTM_HH

#include <math.h>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <stdexcept>

#include "./cell.hh"
#include "./weightsLSTM.hh"
#include "../functions.hh"

class NetworkLSTM {
 public:
    WeightsLSTM* weights;
    int input_size;
    int output_size;
    int layer_size;

    std::vector<Cell> cells;

    NetworkLSTM(WeightsLSTM* weights, int input_size, int output_size, int layer_size);
    std::vector<Eigen::VectorXd> propagate(std::vector<Eigen::VectorXd> inputs);

    void backpropagate(std::vector<Eigen::VectorXd> expected_outputs);
    void reset_layers();
};
#endif
