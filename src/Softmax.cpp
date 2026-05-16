#include "Softmax.h"
#include "Constant.h"

Node SoftmaxGradient::getNewNode(Node &nodeA, Node &nodeB) {
    Node newNode = Node();
    newNode.op = this;
    newNode.input.push_back(nodeA);
    newNode.input.push_back(nodeB);
    newNode.isPlaceHolder = false;
    newNode.name = "SoftmaxGradient(y_predict:" + nodeA.name + ", y_true:" + nodeB.name + ")";
    int hash = 0;
    for (int i = 0; i < (int)newNode.name.length(); i++)
        hash += newNode.name[i];
    newNode.hash_code = hash;
    return newNode;
}

vector<MatrixXd> SoftmaxGradient::compute(Node &nodeA, vector<MatrixXd> &input_vals) {
    vector<MatrixXd> res_mat;
    res_mat.push_back(input_vals[0] - input_vals[1]);
    return res_mat;
}

vector<Node> SoftmaxGradient::gradient(Node &node, Node &output_gradient) {
    return {};
}

SoftmaxGradient softmax_gradient_op = SoftmaxGradient();

Node SoftmaxOp::getNewNode(Node &nodeA, Node &nodeB) {
    Node newNode = Node();
    newNode.op = this;
    newNode.input.push_back(nodeA);
    newNode.input.push_back(nodeB);
    newNode.isPlaceHolder = false;
    newNode.name = "Softmax(" + nodeA.name + ")";
    int hash = 0;
    for (int i = 0; i < (int)newNode.name.length(); i++)
        hash += newNode.name[i];
    newNode.hash_code = hash;
    return newNode;
}

vector<MatrixXd> SoftmaxOp::compute(Node &nodeA, vector<MatrixXd> &input_vals) {
    MatrixXd logits = input_vals[0];
    int rows = logits.rows();
    int cols = logits.cols();
    MatrixXd res_mat(rows, cols);
    for (int i = 0; i < rows; i++) {
        double max_val = logits.row(i).maxCoeff();
        double sum_exp = 0.0;
        for (int j = 0; j < cols; j++) {
            double exp_val = exp(logits(i, j) - max_val);
            res_mat(i, j) = exp_val;
            sum_exp += exp_val;
        }
        if (sum_exp > 0) {
            for (int j = 0; j < cols; j++)
                res_mat(i, j) /= sum_exp;
        }
    }
    vector<MatrixXd> res;
    res.push_back(res_mat);
    nodeA.res = res_mat;
    return res;
}

vector<Node> SoftmaxOp::gradient(Node &node, Node &output_gradient) {
    Node newNodeA = softmax_gradient_op.getNewNode(node, node.input[1]);
    Node newNodeB = zeros_like_op.getNewNode(node.input[1]);
    return {newNodeA, newNodeB};
}

SoftmaxOp softmax_op = SoftmaxOp();
