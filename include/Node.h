#pragma once
#include <iostream>
#include <string>
#include <unordered_set>
#include <map>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>

using namespace Eigen;
using namespace std;

// ============================================================
// Abstract Op base
// ============================================================
class Node;

class Op {
public:
    Op() = default;
    virtual Node getNewNode();
    virtual vector<MatrixXd> compute(Node &node, vector<MatrixXd> &input_vals);
    virtual vector<Node> gradient(Node &node, Node &output_gradient);
    ~Op() = default;
};

// ============================================================
// Graph Node
// ============================================================
class Node {
public:
    Node() {
        this->matmul_attr_trans_A = false;
        this->matmul_attr_trans_B = false;
        this->hash_code = -1;
        this->isPlaceHolder = false;
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

// ============================================================
// Placeholder
// ============================================================
class Placeholders : public Op {
public:
    Placeholders() = default;
    Node getNewNode();
    vector<MatrixXd> compute(Node &nodeA, vector<MatrixXd> &input_vals) override;
    vector<Node> gradient(Node &node, Node &output_gradient) override;
    ~Placeholders() = default;
};

extern Placeholders placeholder_op;

// ============================================================
// Graph utilities
// ============================================================
void topo_sort_dfs(Node &node, unordered_set<int> &visited, vector<Node> &topo_order);
void find_topo_sort(vector<Node> &node_list, vector<Node> &topo_order);
Node sum_node_list(vector<Node> &node_list);
Node Variable(string var_name);
vector<Node> gradients(Node &output_node, vector<Node> &node_list);

// ============================================================
// Executor
// ============================================================
class Executor {
public:
    vector<Node> node_list;
    Executor() = default;
    Executor(vector<Node> eval_node_list);
    ~Executor() = default;
    vector<MatrixXd> run(map<int, MatrixXd> &feed_dic);
};
