// Copyright PinaPL
//
// cell.hh
// PinaPL
//
#ifndef CELL_HH
#define CELL_HH

#include <math.h>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <stdexcept>

#include "./weightsLSTM.hh"

class Cell {
    WeightsLSTM* weights;
    Eigen::VectorXd input;
    Eigen::VectorXd previous_output;
    Eigen::VectorXd previous_cell_state;
    Eigen::VectorXd forget_gate_out;
    Eigen::VectorXd input_gate_out;
    Eigen::VectorXd input_block_out;
    Eigen::VectorXd output_gate_out;
    Eigen::VectorXd cell_state;


 public:
    explicit Cell(WeightsLSTM* weights);
    Eigen::VectorXd cell_out;
    // compute() returns [cell_out, cell_state]
    std::vector<Eigen::VectorXd> compute(
        Eigen::VectorXd previous_output,
        Eigen::VectorXd previous_memory,
        Eigen::VectorXd input);
    // compute_gradient() returns [delta_input, delta_cell_state]
    std::vector<Eigen::VectorXd> compute_gradient(Eigen::VectorXd deltas,
        Eigen::VectorXd previous_delta_cell_in,
        Eigen::VectorXd previous_delta_cell_state);
};
#endif
