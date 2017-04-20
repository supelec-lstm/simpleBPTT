// Copyright PinaPL
//
// layer.hh
// PinaPL
//
#ifndef TEST_HH
#define TEST_HH
#include <stdlib.h>
#include <Eigen/Dense>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include "weights.hh"
#include "network.hh"


void single_cell_test();
void single_grammar_test();
Eigen::VectorXd get_input(char letter);
std::vector<Eigen::VectorXd> real_outputs(std::vector<Eigen::VectorXd> outputs);
std::vector<Eigen::VectorXd> apply_threshold(std::vector<Eigen::VectorXd> real_outputs);
int compare(std::vector<Eigen::VectorXd> real_outputs, std::vector<Eigen::VectorXd> expected_outputs);

#endif
