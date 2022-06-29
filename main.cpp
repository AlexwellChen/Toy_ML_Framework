#include <iostream>
#include <string>
#include <unordered_set>
#include <map>
#include "include/deep_core.h"
#include "include/MatOp.h"
#include <cblas.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

class Node {
public:
    Node() {
        this->const_attr = 0.0;
    }

    vector<Node> input;
    Op *op;
    float const_attr;
    string name;
    int hash_code;
    bool isPlaceHolder;
    bool matmul_attr_trans_A;
    bool matmul_attr_trans_B;

    virtual Node operator*(Node &nodeB);

    virtual Node operator+(Node &nodeB);

    bool operator<(const Node &node) const {
        if (this->hash_code <= node.hash_code)return true;
        if (this->hash_code > node.hash_code)return false;
    }

    ~Node() = default;
};

Node Op::getNewNode() {
    Node newNode = Node();
    newNode.op = this;
    return newNode;
}

vector<MatrixXd> Op::compute(Node &node, vector<MatrixXd> &input_vals) {
    return vector<MatrixXd>();
}

vector<Node> Op::gradient(Node &node, Node &output_gradient) {
    return vector<Node>();
}

class Placeholders : public Op {
public:
    Placeholders() = default;

    Node getNewNode() {
        Node newNode = Node();
        newNode.isPlaceHolder = true;
        newNode.op = this;
        return newNode;
    }

    vector<MatrixXd> compute(Node &node, vector<MatrixXd> &input_vals) override {
        // input_vals[0]代表第一个数，input_vals[1]代表第二个数。
        // 这两个数可以是浮点数，也可以是矩阵。在这里是浮点数，并且存放在长度为1的vector中。
        vector<MatrixXd> res;
        return res;
    }

    vector<Node> gradient(Node &node, Node &output_gradient) override {
        vector<Node> res;
        return res;
    }

    ~Placeholders() = default;
};

class MatAddOp : public Op {
public:


    Node getNewNode(Node &nodeA, Node &nodeB, bool trans_A, bool trans_B) {
        Node newNode = Node();
        newNode.op = this;
        newNode.matmul_attr_trans_A = trans_A;
        newNode.matmul_attr_trans_B = trans_B;
        newNode.input.push_back(nodeA);
        newNode.input.push_back(nodeB);
        newNode.isPlaceHolder = false;
        newNode.name = "MatAdd(" + nodeA.name + ", " + nodeB.name + ") ";
        return newNode;
    }

    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override {
        if (nodeA.matmul_attr_trans_A) {
            input_vals[0].transposeInPlace();
        }
        if (nodeA.matmul_attr_trans_B) {
            input_vals[1].transposeInPlace();
        }
        MatrixXd res_mat = input_vals[0] + input_vals[1];
        vector<MatrixXd> res;
        res.push_back(res_mat);
        return res;
    }

    vector<Node> gradient(Node &node, Node &output_gradient) override {
        vector<Node> res;
        res.push_back(output_gradient);
        res.push_back(output_gradient);
        return res;
    }

    ~MatAddOp() = default;
};

class MatMulOp : public Op {
public:

    Node getNewNode(Node &nodeA, Node &nodeB, bool trans_A, bool trans_B) {
        Node newNode = Node();
        newNode.op = this;
        newNode.matmul_attr_trans_A = trans_A;
        newNode.matmul_attr_trans_B = trans_B;
        newNode.input.push_back(nodeA);
        newNode.input.push_back(nodeB);
        newNode.isPlaceHolder = false;
        newNode.name = "MatMul(" + nodeA.name + ", " + nodeB.name + ")";
        int hash = 0;
        int i = 0;
        for (hash = newNode.name.length(), i = 0; i < newNode.name.length(); i++)
            hash += newNode.name[i];
        newNode.hash_code = hash;
        return newNode;
    }

    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override {
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

    vector<Node> gradient(Node &node, Node &output_gradient) override {
        vector<Node> res;
        Node newNode_A = getNewNode(output_gradient, node.input[1], false, true);
        Node newNode_B = getNewNode(node.input[0], output_gradient, true, false);
        res.push_back(newNode_A);
        res.push_back(newNode_B);
        return res;
    }

    ~MatMulOp() = default;
};

class ReluPrimeOp : public Op {
public:
    ReluPrimeOp() = default;

    Node getNewNode(Node &nodeA, Node &nodeB) {
        Node newNode = Node();
        newNode.op = this;
        newNode.input.push_back(nodeA);
        newNode.input.push_back(nodeB);
        newNode.name = "ReluPrime(" + nodeA.name + ")";
        int hash = 0;
        int i = 0;
        for (hash = newNode.name.length(), i = 0; i < newNode.name.length(); i++)
            hash += newNode.name[i];
        newNode.hash_code = hash;
        newNode.isPlaceHolder = false;
        return newNode;
    }

    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override {
        MatrixXd res_mat(input_vals[0].rows(), input_vals[0].cols());
        for (int i = 0; i < input_vals[0].rows(); ++i)
            for (int j = 0; j < input_vals[0].cols(); ++j) {
                if (input_vals[0](i, j) <= 0) {
                    res_mat(i, j) = 0.0;
                } else res_mat(i, j) = 1.0;
            }
        vector<MatrixXd> res;
        res.push_back(res_mat);
        return res;
    }

    vector<Node> gradient(Node &node, Node &output_gradient) override {
        return vector<Node>(); // No need to implement.
    }
};

ReluPrimeOp relu_prime_op = ReluPrimeOp();

class ReluOp : public Op {
public:
    ReluOp() = default;

    Node getNewNode(Node &nodeA) {
        Node newNode = Node();
        newNode.op = this;
        newNode.input.push_back(nodeA);
        newNode.name = "Relu(" + nodeA.name + ")";
        int hash = 0;
        int i = 0;
        for (hash = newNode.name.length(), i = 0; i < newNode.name.length(); i++)
            hash += newNode.name[i];
        newNode.hash_code = hash;
        newNode.isPlaceHolder = false;
        return newNode;
    }

    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override {
        MatrixXd res_mat(input_vals[0].rows(), input_vals[0].cols());
        for (int i = 0; i < input_vals[0].rows(); ++i)
            for (int j = 0; j < input_vals[0].cols(); ++j) {
                if (input_vals[0](i, j) <= 0) {
                    res_mat(i, j) = 0.0;
                } else res_mat(i, j) = input_vals[0](i, j);
            }
        vector<MatrixXd> res;
        res.push_back(res_mat);
        return res;
    }

    vector<Node> gradient(Node &node, Node &output_gradient) override {
        Node newNode = relu_prime_op.getNewNode(node.input[0], output_gradient);
        vector<Node> res;
        res.push_back(newNode);
        return res;
    }
};
ReluOp relu_op = ReluOp();

class ZerosLikeOp : public Op {
public:
    ZerosLikeOp() = default;

    Node getNewNode(Node &nodeA) {
        Node newNode = Node();
        newNode.op = this;
        newNode.input.push_back(nodeA);
        newNode.name = "Zeroslike(" + nodeA.name + ")";
        int hash = 0;
        int i = 0;
        for (hash = newNode.name.length(), i = 0; i < newNode.name.length(); i++)
            hash += newNode.name[i];
        newNode.hash_code = hash;
        newNode.isPlaceHolder = false;
        return newNode;
    }

    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override {
        MatrixXd zeros = MatrixXd::Zero(input_vals[0].rows(), input_vals[0].cols());
        vector<MatrixXd> res;
        res.push_back(zeros);
        return res;
    }

    vector<Node> gradient(Node &node, Node &output_gradient) override {
        vector<Node> res;
        Node newNode = Node();
        newNode.input.push_back(node.input[0]);
        newNode.name = "Zeroslike(" + node.input[0].name + ")";
        res.push_back(newNode);
        return res;
    }

    ~ZerosLikeOp() = default;
};

class OnesLikeOp : public Op {
public:
    OnesLikeOp() = default;

    Node getNewNode(Node &nodeA) {
        Node newNode = Node();
        newNode.op = this;
        newNode.input.push_back(nodeA);
        newNode.isPlaceHolder = false;
        newNode.name = "Oneslike(" + nodeA.name + ")";
        int hash = 0;
        int i = 0;
        for (hash = newNode.name.length(), i = 0; i < newNode.name.length(); i++)
            hash += newNode.name[i];
        newNode.hash_code = hash;
        return newNode;
    }

    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override {
        MatrixXd ones = MatrixXd::Ones(input_vals[0].rows(), input_vals[0].cols());
        vector<MatrixXd> res;
        res.push_back(ones);
        return res;
    }

    vector<Node> gradient(Node &node, Node &output_gradient) override {
        vector<Node> res;
        Node newNode = Node();
        newNode.input.push_back(node.input[0]);
        newNode.name = "Zeroslike(" + node.input[0].name + ")";
        res.push_back(newNode);
        return res;
    }

    ~OnesLikeOp() = default;
};

Placeholders placeholder_op = Placeholders();
OnesLikeOp ones_like_op = OnesLikeOp();
MatMulOp matmul_op = MatMulOp();
MatAddOp matadd_op = MatAddOp();


Node Variable(string var_name) {
    Node placeholder_node = placeholder_op.getNewNode();
    placeholder_node.name = var_name;
    int hash = 0;
    int i = 0;
    for (hash = var_name.length(), i = 0; i < var_name.length(); i++)
        hash += var_name[i];
    placeholder_node.hash_code = hash;
    return placeholder_node;
}


Node Node::operator*(Node &nodeB) {
    Node nodeA = *this;
    Node newNode = matmul_op.getNewNode(nodeA, nodeB, false, false);
    return newNode;
}

Node Node::operator+(Node &nodeB) {
    Node nodeA = *this;
    Node newNode = matadd_op.getNewNode(nodeA, nodeB, false, false);
    return newNode;
}

void topo_sort_dfs(Node &node, std::unordered_set<int> &visited, vector<Node> &topo_order) {
    if (visited.count(node.hash_code)) {
        return;
    }
    visited.emplace(node.hash_code);
    for (Node &in_node: node.input) {
        topo_sort_dfs(in_node, visited, topo_order);
    }
    topo_order.push_back(node);
}

vector<Node> find_topo_sort(vector<Node> &node_list) {
    std::unordered_set<int> visited;
    vector<Node> topo_order;
    for (Node &node: node_list) {
        topo_sort_dfs(node, visited, topo_order);
    }
    return topo_order;
}

Node sum_node_list(vector<Node> &node_list) {
    Node res = node_list[0];
    int i = 1;
    int size = node_list.size();
    for (i = 1; i < size; i++) {
        res = res + node_list[i];
    }
    return res;
}

class Executor {
public:
    vector<Node> node_list;

    Executor(vector<Node> &eval_node_list) {
        this->node_list = eval_node_list;
    }

    ~Executor() = default;

    vector<MatrixXd> run(map<int, MatrixXd> &feed_dic) {
        vector<Node> topo_order = find_topo_sort(this->node_list);
        vector<MatrixXd> input_vals; // 输入
        vector<MatrixXd> output_vals; // 输出
        for (Node node: topo_order) {
            if (node.isPlaceHolder) {
                continue;
            }
            input_vals.clear();
            for (Node in_node: node.input) {
                input_vals.push_back(feed_dic[in_node.hash_code]);
            }
            auto res = node.op->compute(node, input_vals);
            feed_dic[node.hash_code] = res[0];
        }
        for (Node node: this->node_list) {
            output_vals.push_back(feed_dic[node.hash_code]);
        }
        return output_vals;
    }
};

vector<Node> gradients(Node &output_node, vector<Node> &node_list) {
    map<int, vector<Node>> node_to_output_grads_list;
    vector<Node> ones;
    Node out_ones = ones_like_op.getNewNode(output_node);
    ones.push_back(out_ones);
    node_to_output_grads_list[output_node.hash_code] = ones;

    map<int, Node> node_to_output_grad;

    vector<Node> output_node_list;
    output_node_list.push_back(output_node);
    vector<Node> reverse_topo_sort = find_topo_sort(output_node_list);
    reverse(reverse_topo_sort.begin(), reverse_topo_sort.end());

    for (Node node: reverse_topo_sort) {
        Node grad = sum_node_list(node_to_output_grads_list[node.hash_code]);
        node_to_output_grad[node.hash_code] = grad;
        vector<Node> input_grads = node.op->gradient(node, grad); // 计算当前节点前驱的梯度
        for (int i = 0; i < node.input.size(); i++) {
            node_to_output_grads_list[node.input[i].hash_code] = node_to_output_grads_list[node.input[i].hash_code];
            node_to_output_grads_list[node.input[i].hash_code].push_back(input_grads[i]);
        }
    }
    vector<Node> grad_node_list;
    for (Node node: node_list) {
        grad_node_list.push_back(node_to_output_grad[node.hash_code]);
    }
    return grad_node_list;
}

int main() {
    Node x1 = Variable("x1");
    Node x2 = Variable("x2");
    Node y = x1 * x2;
    Node y1 = relu_op.getNewNode(x1);
    Node y2 = relu_op.getNewNode(y1);

    vector<Node> input_nodes;
    input_nodes.push_back(x1);
//    input_nodes.push_back(y1);
//    input_nodes.push_back(x2);
    vector<Node> grads = gradients(y2, input_nodes);

    vector<Node> exe_list;
    exe_list.push_back(y2);
    exe_list.push_back(grads[0]);
//    exe_list.push_back(grads[1]);
    Executor executor = Executor(exe_list);

    MatrixXd x1_val(3, 2);
//    MatrixXd x2_val(2, 3);
    x1_val << -1, 2, -3, 4, -5, 6; // 3 * 2
//    x2_val << 7, 8, 9, 10, 11, 12; // 2 * 3
    // res: 3 * 3
    map<int, MatrixXd> feed_dic;
    feed_dic[x1.hash_code] = x1_val;
//    feed_dic[x2.hash_code] = x2_val;
    vector<MatrixXd> res = executor.run(feed_dic);
    for (int i = 0; i < exe_list.size(); i++) {
        cout << "Node: " << exe_list[i].name << " value is: " << endl << res[i] << endl;
    }
//    cout << "Node: x1 * x2" << " value is: " << endl << x1_val * x2_val << endl;

}