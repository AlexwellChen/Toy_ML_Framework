#pragma once
#include "Node.h"

class MatAddOp : public Op {
public:
    MatAddOp() = default;
    Node getNewNode(Node &nodeA, Node &nodeB, bool trans_A, bool trans_B);
    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override;
    vector<Node> gradient(Node &node, Node &output_gradient) override;
    ~MatAddOp() = default;
};

extern MatAddOp matadd_op;
