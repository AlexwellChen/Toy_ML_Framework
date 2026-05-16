#include "Constant.h"

// ============================================================
// OnesLikeOp
// ============================================================
Node OnesLikeOp::getNewNode(Node &nodeA) {
    Node newNode;
    newNode.op = this;
    newNode.input.push_back(nodeA);
    newNode.isPlaceHolder = false;
    newNode.name = "Oneslike(" + nodeA.name + ")";
    int hash = 0;
    for (int i = 0; i < (int)newNode.name.length(); i++)
        hash += newNode.name[i];
    newNode.hash_code = hash;
    return newNode;
}

vector<MatrixXd> OnesLikeOp::compute(Node &nodeA, vector<MatrixXd> &input_vals) {
    return {MatrixXd::Ones(input_vals[0].rows(), input_vals[0].cols())};
}

vector<Node> OnesLikeOp::gradient(Node &node, Node &output_gradient) {
    Node newNode;
    newNode.input.push_back(node.input[0]);
    newNode.name = "Zeroslike(" + node.input[0].name + ")";
    int hash = 0;
    for (int i = 0; i < (int)newNode.name.length(); i++)
        hash += newNode.name[i];
    newNode.hash_code = hash;
    return {newNode};
}

OnesLikeOp ones_like_op;

// ============================================================
// ZerosLikeOp
// ============================================================
Node ZerosLikeOp::getNewNode(Node &nodeA) {
    Node newNode;
    newNode.op = this;
    newNode.input.push_back(nodeA);
    newNode.name = "Zeroslike(" + nodeA.name + ")";
    int hash = 0;
    for (int i = 0; i < (int)newNode.name.length(); i++)
        hash += newNode.name[i];
    newNode.hash_code = hash;
    newNode.isPlaceHolder = false;
    return newNode;
}

vector<MatrixXd> ZerosLikeOp::compute(Node &nodeA, vector<MatrixXd> &input_vals) {
    return {MatrixXd::Zero(input_vals[0].rows(), input_vals[0].cols())};
}

vector<Node> ZerosLikeOp::gradient(Node &node, Node &output_gradient) {
    Node newNode;
    newNode.input.push_back(node.input[0]);
    newNode.name = "Zeroslike(" + node.input[0].name + ")";
    int hash = 0;
    for (int i = 0; i < (int)newNode.name.length(); i++)
        hash += newNode.name[i];
    newNode.hash_code = hash;
    return {newNode};
}

ZerosLikeOp zeros_like_op;
