#include "Node.h"
#include "MatMul.h"
#include "MatAdd.h"
#include "Constant.h"
#include <algorithm>

// ============================================================
// Op base
// ============================================================
Node Op::getNewNode() {
    Node newNode;
    newNode.op = this;
    return newNode;
}

vector<MatrixXd> Op::compute(Node &node, vector<MatrixXd> &input_vals) {
    return {};
}

vector<Node> Op::gradient(Node &node, Node &output_gradient) {
    return {};
}

// ============================================================
// Placeholder
// ============================================================
Node Placeholders::getNewNode() {
    Node newNode;
    newNode.isPlaceHolder = true;
    newNode.op = this;
    return newNode;
}

vector<MatrixXd> Placeholders::compute(Node &nodeA, vector<MatrixXd> &input_vals) {
    return {};
}

vector<Node> Placeholders::gradient(Node &node, Node &output_gradient) {
    return {};
}

Placeholders placeholder_op;

// ============================================================
// Node operators
// ============================================================
Node Node::operator*(Node &nodeB) {
    Node nodeA = *this;
    return matmul_op.getNewNode(nodeA, nodeB, false, false);
}

Node Node::operator+(Node &nodeB) {
    Node nodeA = *this;
    return matadd_op.getNewNode(nodeA, nodeB, false, false);
}

// ============================================================
// Graph utilities
// ============================================================
void topo_sort_dfs(Node &node, unordered_set<int> &visited, vector<Node> &topo_order) {
    if (visited.count(node.hash_code)) return;
    visited.emplace(node.hash_code);
    for (Node &in_node: node.input) {
        topo_sort_dfs(in_node, visited, topo_order);
    }
    topo_order.push_back(node);
}

void find_topo_sort(vector<Node> &node_list, vector<Node> &topo_order) {
    unordered_set<int> visited;
    for (Node &node: node_list) topo_sort_dfs(node, visited, topo_order);
}

Node sum_node_list(vector<Node> &node_list) {
    Node res = node_list[0];
    for (int i = 1; i < (int)node_list.size(); i++)
        res = res + node_list[i];
    return res;
}

Node Variable(string var_name) {
    Node placeholder_node = placeholder_op.getNewNode();
    placeholder_node.name = var_name;
    int hash = 0;
    for (int i = 0; i < (int)var_name.length(); i++) hash += var_name[i];
    placeholder_node.hash_code = hash;
    return placeholder_node;
}

vector<Node> gradients(Node &output_node, vector<Node> &node_list) {
    map<int, vector<Node>> node_to_output_grads_list;
    Node out_ones = ones_like_op.getNewNode(output_node);
    node_to_output_grads_list[output_node.hash_code] = {out_ones};

    map<int, Node> node_to_output_grad;
    vector<Node> output_node_list = {output_node};
    vector<Node> reverse_topo_sort;
    find_topo_sort(output_node_list, reverse_topo_sort);
    reverse(reverse_topo_sort.begin(), reverse_topo_sort.end());

    for (Node &node: reverse_topo_sort) {
        Node grad = sum_node_list(node_to_output_grads_list[node.hash_code]);
        node_to_output_grad[node.hash_code] = grad;
        vector<Node> input_grads = node.op->gradient(node, grad);
        for (int i = 0; i < (int)node.input.size(); i++)
            node_to_output_grads_list[node.input[i].hash_code].push_back(input_grads[i]);
    }
    vector<Node> grad_node_list;
    for (Node &node: node_list)
        grad_node_list.push_back(node_to_output_grad[node.hash_code]);
    return grad_node_list;
}

// ============================================================
// Executor
// ============================================================
Executor::Executor(vector<Node> eval_node_list) {
    this->node_list = eval_node_list;
}

vector<MatrixXd> Executor::run(map<int, MatrixXd> &feed_dic) {
    vector<Node> topo_order;
    find_topo_sort(this->node_list, topo_order);
    vector<MatrixXd> input_vals, output_vals;
    for (Node node: topo_order) {
        if (node.isPlaceHolder) continue;
        input_vals.clear();
        for (Node in_node: node.input)
            input_vals.push_back(feed_dic[in_node.hash_code]);
        auto res = node.op->compute(node, input_vals);
        feed_dic[node.hash_code] = res[0];
    }
    for (Node node: this->node_list)
        output_vals.push_back(feed_dic[node.hash_code]);
    return output_vals;
}
