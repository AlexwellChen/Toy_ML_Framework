#pragma once
#include "Node.h"

class OnesLikeOp : public Op {
public:
    OnesLikeOp() = default;
    Node getNewNode(Node &nodeA);
    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override;
    vector<Node> gradient(Node &node, Node &output_gradient) override;
    ~OnesLikeOp() = default;
};

class ZerosLikeOp : public Op {
public:
    ZerosLikeOp() = default;
    Node getNewNode(Node &nodeA);
    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override;
    vector<Node> gradient(Node &node, Node &output_gradient) override;
    ~ZerosLikeOp() = default;
};

extern OnesLikeOp ones_like_op;
extern ZerosLikeOp zeros_like_op;
