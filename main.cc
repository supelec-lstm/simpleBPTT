// Copyright PinaPL
//
// main.cc
// PinaPL
//

#include <eigen3/Eigen/Dense>
#include <string>
#include <fstream>
#include <stdexcept>
#include "weights.hh"
#include "network.hh"
#include "layer.hh"
#include "functions.hh"
#include "iostream"
#include "test.hh"


int main(int argc, char **argv) {
    /*
    int input_size = 5;
    int output_size = 20;
    Weights* weights = new Weights(input_size, output_size);
    Layer* layer = new Layer(weights);


    Eigen::VectorXd input(input_size);
    Eigen::VectorXd previous_outputs(output_size);

    input << 2, 1, -1, 0, 0;
    previous_outputs << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;

    Eigen::VectorXd out = layer->compute(input, previous_outputs);
    std::cout << out << '\n';

    std::cout << "=======================================" << '\n';

    Eigen::VectorXd expected_output(output_size);
    expected_output << 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0;

    Eigen::VectorXd grad = layer->compute_gradient((expected_output - previous_outputs).cwiseProduct(expected_output - previous_outputs));
    std::cout << grad << '\n';

    std::cout << weights->delta_weight_input << '\n';
    std::cout << "applying gradient" << '\n';
    weights->apply_gradient(0.1);

    std::cout << "test cost function" << '\n';
    Eigen::VectorXd test = costfunction(input, previous_outputs);
    std::cout << test << '\n';

    std::cout << "creating network" << '\n';
    Network network = Network(weights, 5, 5, 20);
    std::vector<Eigen::VectorXd> inputs;
    inputs.push_back(input);
    inputs.push_back(input);
    inputs.push_back(input);
    std::cout << "starting propagation" << '\n';
    std::vector<Eigen::VectorXd> propagation = network.propagate(inputs);
    std::cout << "propagation complete" << '\n';
    std::cout << propagation.at(0) << '\n';
    std::cout << propagation.at(1) << '\n';
    std::cout << propagation.at(2) << '\n';
    std::cout << "starting backpropagation" << '\n';
    std::vector<Eigen::VectorXd> expected_outputs;
    expected_outputs.push_back(expected_output);
    expected_outputs.push_back(expected_output);
    expected_outputs.push_back(expected_output);
    network.backpropagate(expected_outputs);
    std::cout << "backpropagation complete" << '\n';
    */
    /*
    Eigen::VectorXd v1(20);
    v1 << 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0;
    Eigen::VectorXd v2(20);
    v2 << 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1;

    std::vector<Eigen::VectorXd> va;
    std::vector<Eigen::VectorXd> vb;

    va.push_back(v1);
    va.push_back(v1);
    va.push_back(v2);
    va.push_back(v1);

    vb.push_back(v2);
    vb.push_back(v2);
    vb.push_back(v2);
    vb.push_back(v2);

    std::cout << "va,va" << '\n';
    std::cout << compare_double(va, va) << '\n';
    std::cout << "va,vb" << '\n';
    std::cout << compare_double(va, vb) << '\n';
    */
    // single_cell_test();
    // single_grammar_test();
    // single_grammar_learn();
    grammar_learn(true);
    return 0;
}
