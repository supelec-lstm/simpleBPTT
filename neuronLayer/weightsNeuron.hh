// Copyright PinaPL
//
// weights.hpp
// PinaPL
//

#ifndef WEIGHTSNEURON_HH
#define WEIGHTSNEURON_HH

#include <eigen3/Eigen/Dense>
#include <stdexcept>

class Weights {
 public:
    int input_size;
    int output_size;
    Weights(int input_size, int output_size);
    ~Weights();
    void apply_gradient(double lambda);


    Eigen::MatrixXd weight_input;
    Eigen::MatrixXd weight_prev;

    Eigen::MatrixXd bias;


    Eigen::MatrixXd delta_weight_input;
    Eigen::MatrixXd delta_weight_prev;

    Eigen::MatrixXd delta_bias;
};
#endif
