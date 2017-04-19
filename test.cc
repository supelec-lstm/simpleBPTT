// Copyright PinaPL
//
// test.cpp
// PinaPL
//

#include <stdlib.h>
#include <Eigen/Dense>
#include <string>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>
#include "test.hh"
#include "weights.hh"
#include "network.hh"

void single_cell_test() {
    int input_size = 26;
    int output_size = 26;
    int layer_size = 30;

    std::cout << "creating weights" << '\n';

    Weights* weights = new Weights(input_size, layer_size);

    std::vector<Eigen::VectorXd> inputs;

    Eigen::VectorXd inputW(input_size);
    inputW << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0;

    Eigen::VectorXd inputA(input_size);
    inputA << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0;

    Eigen::VectorXd inputR(input_size);
    inputR << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0;

    Eigen::VectorXd inputP(input_size);
    inputP << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0;

    Eigen::VectorXd input0(input_size);
    inputP << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0;

    inputs.push_back(inputW);
    inputs.push_back(inputA);
    inputs.push_back(inputR);
    inputs.push_back(inputP);

    std::vector<Eigen::VectorXd> expected_outputs;
    expected_outputs.push_back(inputA);
    expected_outputs.push_back(inputR);
    expected_outputs.push_back(inputP);
    expected_outputs.push_back(input0);

    Network network = Network(weights, input_size, output_size, layer_size);

    std::cout << "Starting learning" << '\n';

    std::vector<Eigen::VectorXd> propagation;

    for (int j=0; j < 1000; j++) {
        std::cout << "Learning no : " << j;
        // std::cout << "Starting propagation" << std::endl;
        std::cout << " - propagation";
        propagation = network.propagate(inputs);
        std::cout << " - backpropagation";
        network.backpropagate(expected_outputs);
        std::cout << " - cleaning" << '\n';
        network.reset_layers();
    }
    std::cout << "Learning Done" << '\n';
    propagation = network.propagate(inputs);
    std::cout << propagation.at(0) << '\n';
    std::cout << "===================" << '\n';
    std::cout << propagation.at(1) << '\n';
    std::cout << "===================" << '\n';
    std::cout << propagation.at(2) << '\n';
    std::cout << "===================" << '\n';
    std::cout << propagation.at(3) << '\n';
    std::cout << "===================" << '\n';
}

/*
void single_cell_grammar_test() {
    int input_size = 7;
    int output_size = 7;
    int words_to_learn = 1;

    Weights* cell_weight = new Weights(input_size, output_size);

    std::ifstream file("reber_test_1M.txt");
    std::string str;
    std::vector<Eigen::MatrixXd> deltas;

    while ((std::getline(file, str)) && (0 < words_to_learn)) {
        int lenght_word = str.length();
        std::vector<Eigen::MatrixXd> deltas;
        std::vector<Eigen::MatrixXd> result;

        Eigen::MatrixXd previous_output =
            Eigen::MatrixXd::Zero(output_size, 1);
        Eigen::MatrixXd previous_memory =
            Eigen::MatrixXd::Zero(output_size, 1);

        std::vector<Cell> network;

        for (int i = 0; i < lenght_word-1; ++i) {
            Cell cell = Cell(cell_weight);
            Eigen::MatrixXd in = get_input(str.at(i));
            Eigen::MatrixXd expected = get_input(str.at(i+1));
            result = cell.compute(previous_output, &previous_memory, in);
            previous_output = result.at(0);
            deltas.push_back((previous_output - expected)
                .cwiseProduct(previous_output - expected));
            previous_memory = result.at(1);
            network.push_back(cell);
        }

        Eigen::MatrixXd previous_delta_cell_in =
            Eigen::MatrixXd::Zero(output_size, 1);
        Eigen::MatrixXd previous_delta_cell_state =
            Eigen::MatrixXd::Zero(output_size, 1);

        for (int i = lenght_word-2; i >= 0; --i) {
            result = network.at(i).compute_gradient(&deltas.at(i),
                &previous_delta_cell_in, &previous_delta_cell_state);
        }
        cell_weight->apply_gradient(0.1);
    }
    std::cout << "Learning done" << std::endl;

    Eigen::MatrixXd previous_output =
        Eigen::MatrixXd::Zero(output_size, 1);
    Eigen::MatrixXd previous_memory =
        Eigen::MatrixXd::Zero(output_size, 1);

    Cell cell = Cell(cell_weight);
    std::vector<Eigen::MatrixXd> result;
    Eigen::MatrixXd B = get_input('B');
    Eigen::MatrixXd P = get_input('P');
    Eigen::MatrixXd V = get_input('V');
    Eigen::MatrixXd E = get_input('E');

    std::cout << "========= On donne B ========" << std::endl;
    result = cell.compute(previous_output, &previous_memory, B);
    previous_output = result.at(0);
    previous_memory = result.at(1);
    std::cout << result.at(0) << std::endl;

    std::cout << "========= On donne P ========" << std::endl;
    result = cell.compute(previous_output, &previous_memory, P);
    previous_output = result.at(0);
    previous_memory = result.at(1);
    std::cout << result.at(0) << std::endl;

    std::cout << "========= On donne V ========" << std::endl;
    result = cell.compute(previous_output, &previous_memory, V);
    previous_output = result.at(0);
    previous_memory = result.at(1);
    std::cout << result.at(0) << std::endl;
}
*/
/*
void single_cell_grammar_test() {
    int input_size = 7;
    int output_size = 7;
    int words_to_learn = 50000;

    Weights* cell_weight = new Weights(input_size, output_size);
    Cell cell = Cell(cell_weight);

    std::ifstream file("reber_test_1M.txt");
    std::string str;
    std::vector<Eigen::MatrixXd> deltas;

    while ((std::getline(file, str)) && (0 < words_to_learn)) {
        int lenght_word = str.length();
        for (int i = 0; i < lenght_word; ++i) {
            Eigen::MatrixXd in = get_input(str.at(i));
            cell.compute(&in);
            deltas.push_back((in - cell.cell_out.back())
                .cwiseProduct((in - cell.cell_out.back())));
        }

        for (int i = lenght_word - 1 ; i >= 0; --i) {
            Eigen::MatrixXd delta = deltas.at(i);
            cell.compute_gate_gradient(&delta, i);
        }
        cell.compute_weight_gradient();
        cell.update_weights(0.3);
        cell.reset();
        words_to_learn -= 1;
    }
    Eigen::MatrixXd in(7, 1);
    in << 1, 0, 0, 0, 0, 0, 0;
    cell.compute(&in);
    std::cout << cell.cell_out.back() << std::endl;
}
*/
Eigen::VectorXd get_input(char letter) {
    Eigen::VectorXd in(7);
    switch (letter) {
        case 'B':
            in << 1, 0, 0, 0, 0, 0, 0;
            break;
        case 'T':
            in << 0, 1, 0, 0, 0, 0, 0;
            break;
        case 'P':
            in << 0, 0, 1, 0, 0, 0, 0;
            break;
        case 'S':
            in << 0, 0, 0, 1, 0, 0, 0;
            break;
        case 'X':
            in << 0, 0, 0, 0, 1, 0, 0;
            break;
        case 'V':
            in << 0, 0, 0, 0, 0, 1, 0;
            break;
        case 'E':
            in << 0, 0, 0, 0, 0, 0, 1;
            break;
    }
    return in;
}
