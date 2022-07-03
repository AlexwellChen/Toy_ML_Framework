#include <iostream>
#include <string>
#include <unordered_set>
#include <map>
#include "include/deep_core.h"
#include "include/MatOp.h"
#include <cblas.h>
#include <Eigen/Dense>

#define BATCH_SIZE 10
#define ITER 100

using namespace Eigen;
using namespace std;

class Node {
public:
    Node() {
        this->const_attr = 0.0;
        this->matmul_attr_trans_A = false;
        this->matmul_attr_trans_B = false;
    }

    vector<Node> input;
    Op *op;
    float const_attr;
    string name;
    int hash_code;
    MatrixXd res;
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

    Node getNewNode() override {
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

Placeholders placeholder_op = Placeholders();

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
        int hash = 0;
        int i = 0;
        for (hash = newNode.name.length(), i = 0; i < newNode.name.length(); i++)
            hash += newNode.name[i];
        newNode.hash_code = hash;
        res.push_back(newNode);
        return res;
    }

    ~OnesLikeOp() = default;
};

OnesLikeOp ones_like_op = OnesLikeOp();

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
        int hash = 0;
        int i = 0;
        for (hash = newNode.name.length(), i = 0; i < newNode.name.length(); i++)
            hash += newNode.name[i];
        newNode.hash_code = hash;
        res.push_back(newNode);
        return res;
    }

    ~ZerosLikeOp() = default;
};

ZerosLikeOp zeros_like_op = ZerosLikeOp();

class MatAddOp : public Op {
public:
    MatAddOp() = default;

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

MatAddOp matadd_op = MatAddOp();

class MatMulOp : public Op {
public:
    MatMulOp() = default;

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
        Node newNodeA = getNewNode(output_gradient, node.input[1], false, true);
        Node newNodeB = getNewNode(node.input[0], output_gradient, true, false);
        res.push_back(newNodeA);
        res.push_back(newNodeB);
        return res;
    }

    ~MatMulOp() = default;
};

MatMulOp matmul_op = MatMulOp();

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

class SoftmaxGradient : public Op {
public:
    SoftmaxGradient() = default;

    Node getNewNode(Node &nodeA, Node &nodeB) {
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

    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override {
        MatrixXd y_hat = input_vals[0]; // Predict
        MatrixXd y = input_vals[1]; // Ground Truth
        vector<MatrixXd> res_mat;
        res_mat.push_back(y_hat - y);
        return res_mat;
    }

    vector<Node> gradient(Node &node, Node &output_gradient) override {
        return vector<Node>();
    }
};

SoftmaxGradient softmax_gradient_op = SoftmaxGradient();

class SoftmaxOp : public Op {
public:
    SoftmaxOp() = default;

    Node getNewNode(Node &nodeA, Node &nodeB) {
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

    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override {
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

    vector<Node> gradient(Node &node, Node &output_gradient) override {
        Node newNodeA = softmax_gradient_op.getNewNode(node, node.input[1]);
        Node newNodeB = zeros_like_op.getNewNode(node.input[1]);
        vector<Node> res_vec;
        res_vec.push_back(newNodeA);
        res_vec.push_back(newNodeB);
        return res_vec;
    }

    ~SoftmaxOp() = default;
};

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

class Executor {
public:
    vector<Node> node_list;

    Executor(vector<Node> &eval_node_list) {
        this->node_list = eval_node_list;
    }

    ~Executor() = default;

    vector<MatrixXd> run(map<int, MatrixXd> &feed_dic) {
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

vector <float> operator/(const vector <float>& m2, const float m1){

    /*  Returns the product of a float and a vectors (elementwise multiplication).
     Inputs:
     m1: float
     m2: vector
     Output: vector, m1 * m2, product of two vectors m1 and m2
     */

    const unsigned long VECTOR_SIZE = m2.size();
    vector <float> product (VECTOR_SIZE);

    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        product[i] = m2[i] / m1;
    };

    return product;
}

void load_data(vector<float> &X_train, vector<float> &y_train){
    string line;
    vector<string> line_v;
    int randindx = rand() % (42000-BATCH_SIZE);
    ifstream myfile ("/Users/alexwell/Desktop/Toy_ML_Framework/train.txt");
    if (myfile.is_open())
    {
        while ( getline (myfile,line) )
        {
            line_v = split(line, '\t');
            int digit = strtof((line_v[0]).c_str(),0);
            for (unsigned i = 0; i < 10; ++i) {
                if (i == digit)
                {
                    y_train.emplace_back(1.);
                }
                else y_train.emplace_back(0.);
            }

            int size = static_cast<int>(line_v.size());
            for (unsigned i = 1; i < size; ++i) {
                X_train.emplace_back(strtof((line_v[i]).c_str(),0));
            }
        }
        X_train = X_train / 255.0;
        myfile.close();
    }
}

void read_batch_data(MatrixXd &input_val, MatrixXd &y_true_val, vector<float> &X_train, vector<float> &y_train){
    int randindx = rand() % (42000-BATCH_SIZE);
    for(unsigned i = randindx*784; i < (randindx+BATCH_SIZE)*784; i += 784){
        for(unsigned j = 0; j < 784; ++j){
            input_val(i / 784 - randindx, j) = X_train[i + j];
        }
    }

    for(unsigned i = randindx*10; i < (randindx+BATCH_SIZE)*10; i += 10){
        for(unsigned j = 0; j < 10; ++j){
            y_true_val(i / 10 - randindx, j) = y_train[i + j];
        }
    }
}

//void print_node_gradients(){
//    map<int, vector<int>>::iterator iter;
//    for(iter = node_to_gradient.begin(); iter != node_to_gradient.end(); iter++) {
//        cout << iter->first << " : ";
//        for(int hash : iter->second){
//            cout << hash << " ";
//        }
//        cout << endl;
//    }
//}

int main() {
    // Softmax test

    map<int, vector<int>> node_to_gradient;
    cout << "3 layers NN test:" << endl;
    Node input = Variable("input");
    Node W1 = Variable("W1");
    Node W2 = Variable("W2");
    Node W3 = Variable("W3");
    Node y_true = Variable("y_true");
    Node input_dot_W1 = input * W1;
    Node W1_relu = relu_op.getNewNode(input_dot_W1);
    Node W1_dot_W2 = W1_relu * W2;
    Node W2_relu = relu_op.getNewNode(W1_dot_W2);
    Node W2_dot_W3 = W2_relu * W3;
    Node y_predict = softmax_op.getNewNode(W2_dot_W3, y_true);

    vector<Node> input_nodes;
    input_nodes.push_back(input);
    input_nodes.push_back(W1);
    input_nodes.push_back(W2);
    input_nodes.push_back(W3);
    input_nodes.push_back(y_true);
    vector<Node> grads = gradients(y_predict, input_nodes);

    node_to_gradient[W1.hash_code].push_back(grads[1].hash_code);
    node_to_gradient[W2.hash_code].push_back(grads[2].hash_code);
    node_to_gradient[W3.hash_code].push_back(grads[3].hash_code);

    vector<Node> exe_list;
    exe_list.push_back(y_predict);
    exe_list.insert(exe_list.end(), grads.begin(), grads.end());
//    exe_list.push_back(grads[0]);
//    exe_list.push_back(grads[1]);
//    exe_list.push_back(grads[2]);
    Executor executor = Executor(exe_list);

    MatrixXd input_val(BATCH_SIZE, 784);
    MatrixXd W1_val = MatrixXd::Random(784,128);
    MatrixXd W2_val = MatrixXd::Random(128,64);
    MatrixXd W3_val = MatrixXd::Random(64,10);
    MatrixXd y_true_val(BATCH_SIZE, 10);

    float lr = .1/BATCH_SIZE;

    vector<float> X_train;
    vector<float> y_train;
    cout << "Start loading data..." << endl;
    load_data(X_train, y_train);
    cout << "Finish loading data..." << endl;

    map<int, MatrixXd> feed_dic;
    feed_dic[W1.hash_code] = W1_val;
    feed_dic[W2.hash_code] = W2_val;
    feed_dic[W3.hash_code] = W3_val;

    int iter_ = 0;
    cout << "------------------------------------" << endl;
    cout << "START TRAINING" << endl;
    cout << "------------------------------------" << endl;
    for(iter_ = 0; iter_ < ITER; iter_++){
        // Read training data
        read_batch_data(input_val, y_true_val, X_train, y_train);
        feed_dic[input.hash_code] = input_val;
        feed_dic[y_true.hash_code] = y_true_val;

        // Train
        vector<MatrixXd> res = executor.run(feed_dic);

        // Weight update
        feed_dic[W1.hash_code] = feed_dic[W1.hash_code] - lr * feed_dic[node_to_gradient[W1.hash_code][0]];
        feed_dic[W2.hash_code] = feed_dic[W2.hash_code] - lr * feed_dic[node_to_gradient[W2.hash_code][0]];
        feed_dic[W3.hash_code] = feed_dic[W3.hash_code] - lr * feed_dic[node_to_gradient[W3.hash_code][0]];

        // Loss
        if(iter_ % 10 == 0){
            MatrixXd loss_m = feed_dic[y_predict.hash_code] - feed_dic[y_true.hash_code];
            cout << "Iteration: " << iter_ <<  ", Loss: " << loss_m.array().square().sum() / (BATCH_SIZE * 10) << endl;
            cout << "------------------------------------" << endl;
        }
    }
}