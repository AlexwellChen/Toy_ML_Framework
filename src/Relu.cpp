#include "Relu.h"

Node ReluPrimeOp::getNewNode(Node &nodeA, Node &nodeB) {
    Node newNode = Node();
    newNode.op = this;
    newNode.input.push_back(nodeA);
    newNode.input.push_back(nodeB);
    newNode.name = "ReluPrime(" + nodeA.name + ")";
    int hash = 0;
    for (int i = 0; i < (int)newNode.name.length(); i++)
        hash += newNode.name[i];
    newNode.hash_code = hash;
    newNode.isPlaceHolder = false;
    return newNode;
}

vector<MatrixXd> ReluPrimeOp::compute(Node &nodeA, vector<MatrixXd> &input_vals) {
    MatrixXd res_mat(input_vals[0].rows(), input_vals[0].cols());
    for (int i = 0; i < input_vals[0].rows(); ++i)
        for (int j = 0; j < input_vals[0].cols(); ++j) {
            if (input_vals[0](i, j) <= 0) res_mat(i, j) = 0.0;
            else res_mat(i, j) = input_vals[1](i, j);
        }
    vector<MatrixXd> res;
    res.push_back(res_mat);
    return res;
}

vector<Node> ReluPrimeOp::gradient(Node &node, Node &output_gradient) {
    return {};
}

ReluPrimeOp relu_prime_op = ReluPrimeOp();

Node ReluOp::getNewNode(Node &nodeA) {
    Node newNode = Node();
    newNode.op = this;
    newNode.input.push_back(nodeA);
    newNode.name = "Relu(" + nodeA.name + ")";
    int hash = 0;
    for (int i = 0; i < (int)newNode.name.length(); i++)
        hash += newNode.name[i];
    newNode.hash_code = hash;
    newNode.isPlaceHolder = false;
    return newNode;
}

vector<MatrixXd> ReluOp::compute(Node &nodeA, vector<MatrixXd> &input_vals) {
    MatrixXd res_mat(input_vals[0].rows(), input_vals[0].cols());
    for (int i = 0; i < input_vals[0].rows(); ++i)
        for (int j = 0; j < input_vals[0].cols(); ++j) {
            if (input_vals[0](i, j) <= 0) res_mat(i, j) = 0.0;
            else res_mat(i, j) = input_vals[0](i, j);
        }
    vector<MatrixXd> res;
    res.push_back(res_mat);
    return res;
}

vector<Node> ReluOp::gradient(Node &node, Node &output_gradient) {
    Node newNode = relu_prime_op.getNewNode(node.input[0], output_gradient);
    return {newNode};
}

ReluOp relu_op = ReluOp();
