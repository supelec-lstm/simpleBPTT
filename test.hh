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

#endif
