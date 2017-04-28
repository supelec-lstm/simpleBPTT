// Copyright PinaPL
//
// cell.cc
// PinaPL
//
#include <math.h>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <stdexcept>
#include "weightsLSTM.hh"
#include "cell.hh"
#include "functions.hh"

Cell::Cell(WeightsLSTM* weights) {
    this->weights = weights;
}

std::vector<Eigen::VectorXd> Cell::compute(
    Eigen::VectorXd previous_output,
    Eigen::VectorXd *previous_memory,
    Eigen::VectorXd input) {
    // TODO : returns [cell_out, cell_state]
}

std::vector<Eigen::VectorXd> Cell::compute_gradient(Eigen::VectorXd* deltas,
    Eigen::VectorXd* previous_delta_cell_in,
    Eigen::VectorXd* previous_delta_cell_state) {
    // TODO : returns [delta_input, delta_cell_state]
}
