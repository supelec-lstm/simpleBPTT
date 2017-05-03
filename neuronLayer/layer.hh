// Copyright PinaPL
//
// layer.hh
// PinaPL
//
#ifndef LAYER_HH
#define LAYER_HH

#include <math.h>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <stdexcept>

#include "./weights.hh"

class Layer {
 public:
    Weights* weights;
    Eigen::VectorXd input;
    Eigen::VectorXd previous_output;
    Eigen::VectorXd output;

    explicit Layer(Weights* weights);
    Eigen::VectorXd compute(Eigen::VectorXd input,
                            Eigen::VectorXd previous_output);

    Eigen::VectorXd compute_gradient(Eigen::VectorXd deltas);
    // void compute_weight_gradient();
};
#endif