// Copyright PinaPL
//
// test.cpp
// PinaPL
//

#include <stdlib.h>
#include <eigen3/Eigen/Dense>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include "test.hh"
#include "weights.hh"
#include "network.hh"

void single_cell_test() {
    int input_size = 26;
    int output_size = 26;
    int layer_size = 40;

    std::cout << "creating weights" << '\n';

    Weights* weights = new Weights(input_size, layer_size);

    std::vector<Eigen::VectorXd> inputs;

    Eigen::VectorXd inputS(input_size);
    inputS << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0;

    Eigen::VectorXd inputH(input_size);
    inputH << 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0;

    Eigen::VectorXd inputA(input_size);
    inputA << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0;

    Eigen::VectorXd inputK(input_size);
    inputK << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0;

    inputs.push_back(inputS);
    inputs.push_back(inputH);
    inputs.push_back(inputA);
    inputs.push_back(inputK);
    inputs.push_back(inputA);
    inputs.push_back(inputS);
    inputs.push_back(inputH);
    inputs.push_back(inputA);
    inputs.push_back(inputK);
    inputs.push_back(inputA);
    inputs.push_back(inputS);
    inputs.push_back(inputH);
    inputs.push_back(inputA);
    inputs.push_back(inputK);
    inputs.push_back(inputA);

    std::vector<Eigen::VectorXd> expected_outputs;
    expected_outputs.push_back(inputH);
    expected_outputs.push_back(inputA);
    expected_outputs.push_back(inputK);
    expected_outputs.push_back(inputA);
    expected_outputs.push_back(inputS);
    expected_outputs.push_back(inputH);
    expected_outputs.push_back(inputA);
    expected_outputs.push_back(inputK);
    expected_outputs.push_back(inputA);
    expected_outputs.push_back(inputS);
    expected_outputs.push_back(inputH);
    expected_outputs.push_back(inputA);
    expected_outputs.push_back(inputK);
    expected_outputs.push_back(inputA);
    expected_outputs.push_back(inputS);


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
        weights->apply_gradient(0.1);
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

void single_grammar_test() {
    int input_size = 7;
    int output_size = 7;
    int layer_size = 25;
    int words_to_learn = 50000;

    Weights* weights = new Weights(input_size, layer_size);
    Network network = Network(weights, input_size, output_size, layer_size);

    std::ifstream file("reber_test_1M.txt");
    std::string str;
    std::vector<Eigen::VectorXd> deltas;
    std::vector<Eigen::VectorXd> inputs;
    std::vector<Eigen::VectorXd> propagation;
    std::vector<Eigen::VectorXd> expected_outputs;


    std::cout << "===== Beginnning of Learning =====" << '\n';
    while ((std::getline(file, str)) && (0 < words_to_learn)) {
        int lenght_word = str.length();
        for (int i = 0; i < lenght_word-1; ++i) {
            inputs.push_back(get_input(str.at(i)));
            expected_outputs.push_back(get_input(str.at(i+1)));
        }

        std::cout << "Words remaining " << words_to_learn;
        std::cout << " - propagation";
        propagation = network.propagate(inputs);
        std::cout << " - backpropagation";
        network.backpropagate(expected_outputs);
        weights->apply_gradient(0.1);
        std::cout << " - cleaning" << '\n';
        network.reset_layers();
        inputs.clear();
        expected_outputs.clear();
        words_to_learn -= 1;
    }
    std::cout << "===== End of Learning =====" << '\n';
    std::cout << "===== Testing =====" << '\n';

    inputs.push_back(get_input('B'));
    inputs.push_back(get_input('P'));
    inputs.push_back(get_input('V'));
    inputs.push_back(get_input('V'));

    propagation = network.propagate(inputs);

    std::cout << "========= On donne B ========" << std::endl;
    std::cout << propagation.at(0) << '\n';

    std::cout << "========= On donne P ========" << std::endl;
    std::cout << propagation.at(1) << '\n';

    std::cout << "========= On donne V ========" << std::endl;
    std::cout << propagation.at(2) << '\n';

    std::cout << "========= On donne V ========" << std::endl;
    std::cout << propagation.at(3) << '\n';
}

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

char get_character(Eigen::VectorXd vector) {
    Eigen::VectorXd B(7);
    B << 1, 0, 0, 0, 0, 0, 0;
    Eigen::VectorXd T(7);
    T << 0, 1, 0, 0, 0, 0, 0;
    Eigen::VectorXd P(7);
    P << 0, 0, 1, 0, 0, 0, 0;
    Eigen::VectorXd S(7);
    S << 0, 0, 0, 1, 0, 0, 0;
    Eigen::VectorXd X(7);
    X << 0, 0, 0, 0, 1, 0, 0;
    Eigen::VectorXd V(7);
    V << 0, 0, 0, 0, 0, 1, 0;
    Eigen::VectorXd E(7);
    E << 0, 0, 0, 0, 0, 0, 1;

    char letter;

    if (vector == B) letter = 'B';
    else if (vector == T) letter = 'T';
    else if (vector == P) letter = 'P';
    else if (vector == S) letter = 'S';
    else if (vector == X) letter = 'X';
    else if (vector == V) letter = 'V';
    else if (vector == E) letter = 'E';
    else letter='*';
    return letter;
}

std::vector<Eigen::VectorXd> real_outputs(std::vector<Eigen::VectorXd> outputs, int end_size) {
    int initial_size = outputs.back().size();
    int lenght_network = outputs.size();
    std::vector<Eigen::VectorXd> real_output;
    for (int i = 0; i < lenght_network; i++) {
        real_output.push_back(Eigen::MatrixXd::Identity(end_size, initial_size) * outputs.at(i));
    }
    return(real_output);
}

std::vector<Eigen::VectorXd> apply_threshold(std::vector<Eigen::VectorXd> real_outputs) {
    for (size_t i = 0; i < real_outputs.size(); i++) {
        real_outputs.at(i) = real_outputs.at(i).unaryExpr(&threshold);
    }
    return(real_outputs);
}

int compare(std::vector<Eigen::VectorXd> real_outputs,
            std::vector<Eigen::VectorXd> expected_outputs) {
    int score = 0;
    Eigen::VectorXd diff;
    int size = real_outputs.size();
    bool transition_predicted;
    // for each VectorXd
    for (int i = 0; i < size; i++) {
        // We compare the state predicted and the next state
        diff = real_outputs.at(i) - expected_outputs.at(i);
        transition_predicted = true;
        for (int j = 0; j < diff.size(); j++) {
            // if one of the coordinates is <0 there is a transition not predicted
            if (diff(j) < 0) {
                transition_predicted = false;
            }
        }
        // If we did not found any error, we score
        if (transition_predicted) score+=1;
    }
    // Checks if we preditected ALL the transitions
    if (score == size) {
        return(1);
    } else {
        return(0);
    }
}

void grammar_learn(bool dual) {
    int input_size = 7;
    int output_size = 7;
    int layer_size = 30;
    int batch_to_learn = 10000;
    int batch_size = 10;
    int current_batch_size;
    int offset;

    // time seed
    srand(time(NULL));

    // random offset in data set
    offset = rand() % 100000;

    Weights* weights = new Weights(input_size, layer_size);
    Network network = Network(weights, input_size, output_size, layer_size);

    std::ifstream file(open_file(dual));
    std::string str;
    std::vector<Eigen::VectorXd> deltas;
    std::vector<Eigen::VectorXd> inputs;
    std::vector<Eigen::VectorXd> propagation;
    std::vector<Eigen::VectorXd> expected_outputs;

    // std::cout << "===== Beginnning of Learning =====" << '\n';
    for (int batch = 0; batch < batch_to_learn; batch++) {
        // std::cout << "batch no"<< batch;
        std::cout << batch;
        current_batch_size = batch_size;
        while ((std::getline(file, str)) && (0 < current_batch_size)) {
            int lenght_word = str.length();
            for (int i = 0; i < lenght_word-1; ++i) {
                inputs.push_back(get_input(str.at(i)));
                expected_outputs.push_back(get_input(str.at(i+1)));
            }
            propagation = network.propagate(inputs);
            network.backpropagate(expected_outputs);
            network.reset_layers();
            inputs.clear();
            expected_outputs.clear();
            weights->apply_gradient(0.1);
            current_batch_size -= 1;
        }
        if (dual) {
            double_grammar_evaluate(network, 1000);
        } else {
            single_grammar_evaluate(network, 1000);
        }
    }
}
void single_grammar_evaluate(Network network, int words_to_test) {
    std::ifstream file("reber_test_1M.txt");
    std::string str;
    std::vector<Eigen::VectorXd> inputs;
    std::vector<Eigen::VectorXd> propagation;
    std::vector<Eigen::VectorXd> expected_outputs;
    int score = 0;
    int remaining_words_to_test = words_to_test;

    // We add an offset
    int offset;

    // time seed
    srand(time(NULL));

    offset = rand() % 10000;  // between 0 and 9999
    for (int i = 0; i < offset; i++) {
        std::getline(file, str);  // dirty way of skipping lines
    }

    std::cout << ",";
    while ((std::getline(file, str)) && (0 < remaining_words_to_test)) {
        int lenght_word = str.length();
        for (int i = 0; i < lenght_word-1; ++i) {
            inputs.push_back(get_input(str.at(i)));
            expected_outputs.push_back(get_input(str.at(i+1)));
        }
        propagation = network.propagate(inputs);
        network.reset_layers();
        inputs.clear();
        score += compare(apply_threshold(real_outputs(propagation, network.output_size)),
                         expected_outputs);
        expected_outputs.clear();
        remaining_words_to_test -= 1;
    }
    float score_percent = (float) 100 * score / words_to_test;
    // std::cout << "score :" << score << '\n';
    std::cout << score_percent << '\n';
}

void double_grammar_evaluate(Network network, int words_to_test) {
    // Here we test the symmetrical reber grammar
    // Initialisation bloc
    std::ifstream file("symmetrical_reber_test_1M.txt");
    std::string str;
    std::vector<Eigen::VectorXd> inputs;
    std::vector<Eigen::VectorXd> propagation;
    std::vector<Eigen::VectorXd> expected_outputs;

    // We add an offset
    int offset;

    // time seed
    srand(time(NULL));

    offset = rand() % 10000;  // between 0 and 9999
    for (int i = 0; i < offset; i++) {
        std::getline(file, str);  // dirty way of skipping lines
    }

    int score = 0;

    // While we have words to test
    while ((std::getline(file, str)) && (0 < words_to_test)) {
        int lenght_word = str.length();
        // We read each letter
        for (int i = 0; i < lenght_word-1; ++i) {
            // We populate the inputs and outputs datasets
            inputs.push_back(get_input(str.at(i)));
            expected_outputs.push_back(get_input(str.at(i+1)));
        }

        propagation = network.propagate(inputs);

        // We clean everything just after
        network.reset_layers();
        inputs.clear();
        score += compare_double(apply_threshold(real_outputs(propagation, network.output_size)),
                                expected_outputs);
        expected_outputs.clear();
        words_to_test -= 1;
    }
    std::cout << " - score :" << score << '\n';
}

int compare_double(std::vector<Eigen::VectorXd> real_outputs,
                   std::vector<Eigen::VectorXd> expected_outputs) {
    int score = 0;
    Eigen::VectorXd diff;
    int size = real_outputs.size();
    bool transition_predicted;

    // We compare the last state predicted and the first transition
    // std::cout << "real - expected" << '\n';
    // std::cout << get_character(real_outputs.at(size-2)) << " - " << get_character(expected_outputs.at(size-2)) << '\n';
    diff = real_outputs.at(size-2) - expected_outputs.at(size-2);
    transition_predicted = true;
    for (size_t j = 0; j < diff.size(); j++) {
        // if one of the coordinates is <0 there is a transition not predicted
        if (std::abs(diff(j)) > 0.1) {
            transition_predicted = false;
        }
    }
        // If we did not found any error, we score
    if (transition_predicted) score=1;
    return(score);
}
