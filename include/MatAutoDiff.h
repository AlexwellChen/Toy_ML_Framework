#include <iostream>
#include <string>
#include <unordered_set>
#include <map>
#include "MatOp.h"
#include <vector>
#include <Eigen/Dense>

class Node {
public:
    Node() {
        this->matmul_attr_trans_A = false;
        this->matmul_attr_trans_B = false;
    }

    vector<Node> input;
    Op *op;
    string name;
    int hash_code;
    MatrixXd res;
    bool isPlaceHolder;
    bool matmul_attr_trans_A;
    bool matmul_attr_trans_B;

    virtual Node operator*(Node &nodeB);

    virtual Node operator+(Node &nodeB);

    ~Node() = default;
};

void topo_sort_dfs(Node &node, std::unordered_set<int> &visited, vector<Node> &topo_order);
void find_topo_sort(vector<Node> &node_list, vector<Node> &topo_order);
Node sum_node_list(vector<Node> &node_list);

class Executor {
public:
    vector<Node> node_list;

    Executor(vector<Node> &eval_node_list);

    ~Executor() = default;

    vector<MatrixXd> run(map<int, MatrixXd> &feed_dic);

};

class Placeholders : public Op {
public:
    Node getNewNode();

    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override;

    vector<Node> gradient(Node &node, Node &output_gradient) override;

    Placeholders() = default;

    ~Placeholders() = default;
};

class ReluPrimeOp : public Op {
public:
    ReluPrimeOp() = default;

    Node getNewNode(Node &nodeA, Node &nodeB);

    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override;

    vector<Node> gradient(Node &node, Node &output_gradient) override;

    ~ReluPrimeOp() = default;
};

class ReluOp : public Op {
public:
    ReluOp() = default;

    Node getNewNode(Node &nodeA);

    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override;

    vector<Node> gradient(Node &node, Node &output_gradient) override;

};

class SoftmaxGradient : public Op {
public:
    SoftmaxGradient() = default;

    Node getNewNode(Node &nodeA, Node &nodeB);

    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override;

    vector<Node> gradient(Node &node, Node &output_gradient) override;
};

class SoftmaxOp : public Op {
public:
    SoftmaxOp() = default;

    Node getNewNode(Node &nodeA, Node &nodeB);

    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override;

    vector<Node> gradient(Node &node, Node &output_gradient) override;

    ~SoftmaxOp() = default;
};

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

class MatMulOp : public Op {
public:
    MatMulOp() = default;

    Node getNewNode(Node &nodeA, Node &nodeB, bool trans_A, bool trans_B);

    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override;

    vector<Node> gradient(Node &node, Node &output_gradient) override;

    ~MatMulOp() = default;
};

class MatAddOp : public Op {
public:
    MatAddOp() = default;

    Node getNewNode(Node &nodeA, Node &nodeB, bool trans_A, bool trans_B);

    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override;

    vector<Node> gradient(Node &node, Node &output_gradient) override;

    ~MatAddOp() = default;
};

Node Variable(string var_name);
vector<Node> gradients(Node &output_node, vector<Node> &node_list);
void load_data(vector<float> &X_train, vector<float> &y_train);
void read_batch_data(MatrixXd &input_val, MatrixXd &y_true_val, vector<float> &X_train, vector<float> &y_train);