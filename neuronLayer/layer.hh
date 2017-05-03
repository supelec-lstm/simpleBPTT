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

#include "./weightsNeuron.hh"

class Layer {
 public:
    WeightsNeuron* weights;
    Eigen::VectorXd input;
    Eigen::VectorXd previous_output;
    Eigen::VectorXd output;

    explicit Layer(WeightsNeuron* weights);
    Eigen::VectorXd compute(Eigen::VectorXd input,
                            Eigen::VectorXd previous_output);

    Eigen::VectorXd compute_gradient(Eigen::VectorXd deltas);
    // void compute_weight_gradient();
};
#endif
