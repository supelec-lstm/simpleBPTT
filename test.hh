// Copyright PinaPL
//
// layer.hh
// PinaPL
//
#ifndef TEST_HH
#define TEST_HH
#include <stdlib.h>
#include <eigen3/Eigen/Dense>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include "neuronLayer/weightsNeuron.hh"
#include "neuronLayer/networkNeuron.hh"


void single_cell_test();
void single_grammar_test();
Eigen::VectorXd get_input(char letter);
std::vector<Eigen::VectorXd> real_outputs(std::vector<Eigen::VectorXd> outputs);
std::vector<Eigen::VectorXd> apply_threshold(std::vector<Eigen::VectorXd> real_outputs);

int compare(std::vector<Eigen::VectorXd> real_outputs,
            std::vector<Eigen::VectorXd> expected_outputs);

int compare_double(std::vector<Eigen::VectorXd> propagation,
                   std::vector<Eigen::VectorXd> expected_outputs);

void grammar_learn(bool symmetrical, bool lstm);

void single_grammar_evaluate(Network network, int words_to_test);

void double_grammar_evaluate(Network network, int words_to_test);

std::vector<Eigen::VectorXd> real_outputs(std::vector<Eigen::VectorXd> outputs, int end_size);

#endif
