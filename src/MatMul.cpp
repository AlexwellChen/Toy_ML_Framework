#include "MatMul.h"

Node MatMulOp::getNewNode(Node &nodeA, Node &nodeB, bool trans_A, bool trans_B) {
    Node newNode = Node();
    newNode.op = this;
    newNode.matmul_attr_trans_A = trans_A;
    newNode.matmul_attr_trans_B = trans_B;
    newNode.input.push_back(nodeA);
    newNode.input.push_back(nodeB);
    newNode.isPlaceHolder = false;
    newNode.name = "MatMul(" + nodeA.name + ", " + nodeB.name + ")";
    int hash = 0;
    for (int i = 0; i < (int)newNode.name.length(); i++)
        hash += newNode.name[i];
    newNode.hash_code = hash;
    return newNode;
}

vector<MatrixXd> MatMulOp::compute(Node &nodeA, vector<MatrixXd> &input_vals) {
    MatrixXd res_mat;
    if (nodeA.matmul_attr_trans_A && nodeA.matmul_attr_trans_B) {
        res_mat = input_vals[0].transpose() * input_vals[1].transpose();
    } else if (!nodeA.matmul_attr_trans_A && nodeA.matmul_attr_trans_B) {
        res_mat = input_vals[0] * input_vals[1].transpose();
    } else if (nodeA.matmul_attr_trans_A && !nodeA.matmul_attr_trans_B) {
        res_mat = input_vals[0].transpose() * input_vals[1];
    } else {
        res_mat = input_vals[0] * input_vals[1];
    }
    vector<MatrixXd> res;
    res.push_back(res_mat);
    return res;
}

vector<Node> MatMulOp::gradient(Node &node, Node &output_gradient) {
    vector<Node> res;
    Node newNodeA = getNewNode(output_gradient, node.input[1], false, true);
    Node newNodeB = getNewNode(node.input[0], output_gradient, true, false);
    res.push_back(newNodeA);
    res.push_back(newNodeB);
    return res;
}

MatMulOp matmul_op = MatMulOp();
