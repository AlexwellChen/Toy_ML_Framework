#include <iostream>
#include <string>
#include <unordered_set>
#include <map>
#include "../include/MatAutoDiff.h"
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <omp.h>

using namespace Eigen;
using namespace std;

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

Node Placeholders::getNewNode() {
    Node newNode = Node();
    newNode.isPlaceHolder = true;
    newNode.op = this;
    return newNode;
}

vector<MatrixXd> Placeholders::compute(Node &node, vector<MatrixXd> &input_vals) {
    // input_vals[0]代表第一个数，input_vals[1]代表第二个数。
    // 这两个数可以是浮点数，也可以是矩阵。在这里是浮点数，并且存放在长度为1的vector中。
    vector<MatrixXd> res;
    return res;
}

vector<Node> Placeholders::gradient(Node &node, Node &output_gradient) {
    vector<Node> res;
    return res;
}

Placeholders placeholder_op = Placeholders();

Node OnesLikeOp::getNewNode(Node &nodeA) {
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

vector<MatrixXd> OnesLikeOp::compute(Node &nodeA, vector<MatrixXd> &input_vals) {
    MatrixXd ones = MatrixXd::Ones(input_vals[0].rows(), input_vals[0].cols());
    vector<MatrixXd> res;
    res.push_back(ones);
    return res;
}

vector<Node> OnesLikeOp::gradient(Node &node, Node &output_gradient) {
    vector<Node> res;
    Node newNode = Node();
    newNode.input.push_back(node.input[0]);
    newNode.name = "Zeroslike(" + node.input[0].name + ")";
    int hash = 0;
    int i = 0;
    for (hash = newNode.name.length(), i = 0; i < newNode.name.length(); i++)
        hash += newNode.name[i];
    newNode.hash_code = hash;
    res.push_back(newNode);
    return res;
}

OnesLikeOp ones_like_op = OnesLikeOp();

Node ZerosLikeOp::getNewNode(Node &nodeA) {
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

vector<MatrixXd> ZerosLikeOp::compute(Node &nodeA, vector<MatrixXd> &input_vals) {
    MatrixXd zeros = MatrixXd::Zero(input_vals[0].rows(), input_vals[0].cols());
    vector<MatrixXd> res;
    res.push_back(zeros);
    return res;
}

vector<Node> ZerosLikeOp::gradient(Node &node, Node &output_gradient) {
    vector<Node> res;
    Node newNode = Node();
    newNode.input.push_back(node.input[0]);
    newNode.name = "Zeroslike(" + node.input[0].name + ")";
    int hash = 0;
    int i = 0;
    for (hash = newNode.name.length(), i = 0; i < newNode.name.length(); i++)
        hash += newNode.name[i];
    newNode.hash_code = hash;
    res.push_back(newNode);
    return res;
}

ZerosLikeOp zeros_like_op = ZerosLikeOp();

Node MatAddOp::getNewNode(Node &nodeA, Node &nodeB, bool trans_A, bool trans_B) {
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

vector<MatrixXd> MatAddOp::compute(Node &nodeA, vector<MatrixXd> &input_vals) {
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

vector<Node> MatAddOp::gradient(Node &node, Node &output_gradient) {
    vector<Node> res;
    res.push_back(output_gradient);
    res.push_back(output_gradient);
    return res;
}

MatAddOp matadd_op = MatAddOp();

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
    int i = 0;
    for (hash = newNode.name.length(), i = 0; i < newNode.name.length(); i++)
        hash += newNode.name[i];
    newNode.hash_code = hash;
    return newNode;
}

vector<MatrixXd> MatMulOp::compute(Node &nodeA, vector<MatrixXd> &input_vals) {
    MatrixXd res_mat;
    omp_set_num_threads(4);
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

Node ReluPrimeOp::getNewNode(Node &nodeA, Node &nodeB) {
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

vector<MatrixXd> ReluPrimeOp::compute(Node &nodeA, vector<MatrixXd> &input_vals) {
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

vector<Node> ReluPrimeOp::gradient(Node &node, Node &output_gradient) {
    return {}; // No need to implement.
}

ReluPrimeOp relu_prime_op = ReluPrimeOp();

Node ReluOp::getNewNode(Node &nodeA) {
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

vector<MatrixXd> ReluOp::compute(Node &nodeA, vector<MatrixXd> &input_vals) {
    MatrixXd res_mat(input_vals[0].rows(), input_vals[0].cols());
#pragma omp parallel for
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

vector<Node> ReluOp::gradient(Node &node, Node &output_gradient) {
    Node newNode = relu_prime_op.getNewNode(node.input[0], output_gradient);
    vector<Node> res;
    res.push_back(newNode);
    return res;
}

ReluOp relu_op = ReluOp();

Node SoftmaxGradient::getNewNode(Node &nodeA, Node &nodeB) {
    Node newNode = Node();
    newNode.op = this;
    newNode.input.push_back(nodeA);
    newNode.input.push_back(nodeB);
    newNode.isPlaceHolder = false;
    newNode.name = "SoftmaxGradient(y_predict:" + nodeA.name + ", y_true:" + nodeB.name + ")";
    int hash = 0;
    int i = 0;
    for (hash = newNode.name.length(), i = 0; i < newNode.name.length(); i++)
        hash += newNode.name[i];
    newNode.hash_code = hash;
    newNode.isPlaceHolder = false;
    return newNode;
}

vector<MatrixXd> SoftmaxGradient::compute(Node &nodeA, vector<MatrixXd> &input_vals) {
    MatrixXd y_hat = input_vals[0]; // Predict
    MatrixXd y = input_vals[1]; // Ground Truth
    vector<MatrixXd> res_mat;
    res_mat.push_back(y_hat - y);
    return res_mat;
}

vector<Node> SoftmaxGradient::gradient(Node &node, Node &output_gradient) {
    return vector<Node>();
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
    int i = 0;
    for (hash = newNode.name.length(), i = 0; i < newNode.name.length(); i++)
        hash += newNode.name[i];
    newNode.hash_code = hash;
    newNode.isPlaceHolder = false;
    return newNode;
}

vector<MatrixXd> SoftmaxOp::compute(Node &nodeA, vector<MatrixXd> &input_vals) {
    MatrixXd m = input_vals[0].array().exp();
    MatrixXd sum = m.colwise().sum();
    MatrixXd res_mat(m.rows(), m.cols());
    for (int i = 0; i < m.rows(); i++)
        for (int j = 0; j < m.cols(); j++)
            res_mat(i, j) = m(i, j) / sum(0, j);
    vector<MatrixXd> res;
    res.push_back(res_mat);
    nodeA.res = res_mat;
    return res;
}

vector<Node> SoftmaxOp::gradient(Node &node, Node &output_gradient) {
    Node newNodeA = softmax_gradient_op.getNewNode(node, node.input[1]);
    Node newNodeB = zeros_like_op.getNewNode(node.input[1]);
    vector<Node> res_vec;
    res_vec.push_back(newNodeA);
    res_vec.push_back(newNodeB);
    return res_vec;
}

SoftmaxOp softmax_op = SoftmaxOp();

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

void find_topo_sort(vector<Node> &node_list, vector<Node> &topo_order) {
    std::unordered_set<int> visited;
    for (Node &node: node_list) {
        topo_sort_dfs(node, visited, topo_order);
    }
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


vector<Node> gradients(Node &output_node, vector<Node> &node_list) {
    map<int, vector<Node>> node_to_output_grads_list;
    vector<Node> ones;
    Node out_ones = ones_like_op.getNewNode(output_node);
    ones.push_back(out_ones);
    node_to_output_grads_list[output_node.hash_code] = ones;

    map<int, Node> node_to_output_grad;

    vector<Node> output_node_list;
    output_node_list.push_back(output_node);
    vector<Node> reverse_topo_sort;
    find_topo_sort(output_node_list, reverse_topo_sort);
    reverse(reverse_topo_sort.begin(), reverse_topo_sort.end());

    for (Node &node: reverse_topo_sort) {
        Node grad = sum_node_list(node_to_output_grads_list[node.hash_code]);
        node_to_output_grad[node.hash_code] = grad;
        vector<Node> input_grads = node.op->gradient(node, grad); // 计算当前节点前驱的梯度
        for (int i = 0; i < node.input.size(); i++) {
            node_to_output_grads_list[node.input[i].hash_code] = node_to_output_grads_list[node.input[i].hash_code];
            node_to_output_grads_list[node.input[i].hash_code].push_back(input_grads[i]);
        }
    }
    vector<Node> grad_node_list;
    for (Node &node: node_list) {
        grad_node_list.push_back(node_to_output_grad[node.hash_code]);
    }
    return grad_node_list;
}

vector<string> split(const string &s, char delim) {
    stringstream ss(s);
    string item;
    vector<string> tokens;
    while (getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}

vector<float> operator/(const vector<float> &m2, const float m1) {

    /*  Returns the product of a float and a vectors (elementwise multiplication).
     Inputs:
     m1: float
     m2: vector
     Output: vector, m1 * m2, product of two vectors m1 and m2
     */

    const unsigned long VECTOR_SIZE = m2.size();
    vector<float> product(VECTOR_SIZE);

    for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
        product[i] = m2[i] / m1;
    };

    return product;
}

void load_data(vector<float> &X_train, vector<float> &y_train, string src) {
    string line;
    vector<string> line_v;
    ifstream myfile(src);
    if (myfile.is_open()) {
        while (getline(myfile, line)) {
            line_v = split(line, '\t');
            int digit = strtof((line_v[0]).c_str(), 0);
            for (unsigned i = 0; i < 10; ++i) {
                if (i == digit) {
                    y_train.emplace_back(1.);
                } else y_train.emplace_back(0.);
            }

            int size = static_cast<int>(line_v.size());
            for (unsigned i = 1; i < size; ++i) {
                X_train.emplace_back(strtof((line_v[i]).c_str(), 0));
            }
        }
        X_train = X_train / 255.0;
        myfile.close();
    }
}

void read_batch_data(MatrixXd &input_val, MatrixXd &y_true_val, vector<float> &X_train, vector<float> &y_train,
                     int batch_size) {

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<int> dist(0, 42000 * 10);
    int rand_indx = dist(mt) % (42000 - batch_size);
    for (unsigned i = rand_indx * 784; i < (rand_indx + batch_size) * 784; i += 784) {
        for (unsigned j = 0; j < 784; ++j) {
            input_val(i / 784 - rand_indx, j) = X_train[i + j];
        }
    }
    for (unsigned i = rand_indx * 10; i < (rand_indx + batch_size) * 10; i += 10) {
        for (unsigned j = 0; j < 10; ++j) {
            y_true_val(i / 10 - rand_indx, j) = y_train[i + j];
        }
    }
}


Executor::Executor(vector<Node> &eval_node_list) {
    this->node_list = eval_node_list;
}

vector<MatrixXd> Executor::run(map<int, MatrixXd> &feed_dic) {
    vector<Node> topo_order;
    find_topo_sort(this->node_list, topo_order);
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