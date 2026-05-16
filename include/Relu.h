#pragma once
#include "Node.h"

class ReluPrimeOp : public Op {
public:
    ReluPrimeOp() = default;
    Node getNewNode(Node &nodeA, Node &nodeB);
    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override;
    vector<Node> gradient(Node &node, Node &output_gradient) override;
    ~ReluPrimeOp() = default;
};

class ReluOp : public Op {
public:
    ReluOp() = default;
    Node getNewNode(Node &nodeA);
    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override;
    vector<Node> gradient(Node &node, Node &output_gradient) override;
    ~ReluOp() = default;
};

extern ReluPrimeOp relu_prime_op;
extern ReluOp relu_op;
