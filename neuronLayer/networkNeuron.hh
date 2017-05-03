// Copyright PinaPL
//
// layer.hh
// PinaPL
//
#ifndef NETWORKNEURON_HH
#define NETWORKNEURON_HH

#include <math.h>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <stdexcept>

#include "./layer.hh"
#include "../functions.hh"

class Network {
 public:
    Weights* weights;
    int input_size;
    int output_size;
    int layer_size;

    std::vector<Layer> layers;

    Network(Weights* weights, int input_size, int output_size, int layer_size);
    std::vector<Eigen::VectorXd> propagate(std::vector<Eigen::VectorXd> inputs);

    void backpropagate(std::vector<Eigen::VectorXd> expected_outputs);
    void reset_layers();
};
#endif
