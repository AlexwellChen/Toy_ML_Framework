#pragma once
#include "Node.h"

class SoftmaxGradient : public Op {
public:
    SoftmaxGradient() = default;
    Node getNewNode(Node &nodeA, Node &nodeB);
    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override;
    vector<Node> gradient(Node &node, Node &output_gradient) override;
};

class SoftmaxOp : public Op {
public:
    SoftmaxOp() = default;
    Node getNewNode(Node &nodeA, Node &nodeB);
    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override;
    vector<Node> gradient(Node &node, Node &output_gradient) override;
    ~SoftmaxOp() = default;
};

extern SoftmaxGradient softmax_gradient_op;
extern SoftmaxOp softmax_op;
