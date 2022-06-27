//
// Created by 陈方舟 on 2022/6/27.
//

#ifndef TOY_ML_FRAMEWORK_OP_H
#define TOY_ML_FRAMEWORK_OP_H

#endif //TOY_ML_FRAMEWORK_OP_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <unordered_set>


using namespace std;

class Node;
class Op {
public:
    Op() = default;

    virtual Node getNewNode();

    virtual vector<vector<float>> compute(Node &node, vector<vector<float>> &input_vals);

    virtual vector<Node> gradient(Node &node, Node &output_gradient);

    ~Op() = default;
};

