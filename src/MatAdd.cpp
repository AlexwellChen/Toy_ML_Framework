#include "MatAdd.h"

Node MatAddOp::getNewNode(Node &nodeA, Node &nodeB, bool trans_A, bool trans_B) {
    Node newNode = Node();
    newNode.op = this;
    newNode.matmul_attr_trans_A = trans_A;
    newNode.matmul_attr_trans_B = trans_B;
    newNode.input.push_back(nodeA);
    newNode.input.push_back(nodeB);
    newNode.isPlaceHolder = false;
    newNode.name = "MatAdd(" + nodeA.name + ", " + nodeB.name + ") ";
    int hash = 0;
    for (int i = 0; i < (int)newNode.name.length(); i++)
        hash += newNode.name[i];
    newNode.hash_code = hash;
    return newNode;
}

vector<MatrixXd> MatAddOp::compute(Node &nodeA, vector<MatrixXd> &input_vals) {
    MatrixXd a = input_vals[0], b = input_vals[1];
    if (a.rows() == 1 && b.rows() > 1) {
        a = a.replicate(b.rows(), 1);
    } else if (b.rows() == 1 && a.rows() > 1) {
        b = b.replicate(a.rows(), 1);
    }
    vector<MatrixXd> res;
    res.push_back(a + b);
    return res;
}

vector<Node> MatAddOp::gradient(Node &node, Node &output_gradient) {
    vector<Node> res;
    res.push_back(output_gradient);
    res.push_back(output_gradient);
    return res;
}

MatAddOp matadd_op = MatAddOp();
