#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <unordered_set>
#include <map>
#include "include/deep_core.h"
#include "include/vector_ops.h"

using namespace std;
//
//class Op {
//public:
//    Op() = default;
//
//    Node getNewNode() {
//        Node newNode = Node();
//        newNode.op = this;
//        return newNode;
//    }
//
//    virtual vector<vector<float>> compute(Node &node, vector<vector<float>> &input_vals);
//
//    virtual vector<float> gradient(Node &node, float &output_gradient);
//
//    ~Op() {};
//};

class Node {
public:
    Node(){
        int hash, i;
        for (hash = this->name.length(), i = 0; i < this->name.length(); i++)
            hash += this->name[i];
        this->hash_code = (hash % 31);
        this->const_attr = 0.0;
    }
    vector<Node> input;
//    Op *op = 0;
    float const_attr;
    string name;
    int hash_code;
    bool isPlaceHolder;

    virtual vector<vector<float>> compute(Node &node, vector<vector<float>> &input_vals);

    virtual vector<Node> gradient(Node &node, vector<Node> &output_gradient);

    virtual Node operator+(float &value);

    virtual Node operator+(Node &nodeB);

    string getName(){
        return this->name;
    }

    bool operator<(const Node& node) const{
        if (this->hash_code <= node.hash_code)return true;
        if (this->hash_code > node.hash_code)return false;
    }

    ~Node() = default;

};

class hash_fun {
public:
    int operator()(const Node &A) const {
        int hash, i;
//        string key = A.name;
        for (hash = A.name.length(), i = 0; i < A.name.length(); i++)
            hash += A.name[i];
        return (hash % 31);
    }
};

class hash_cmp {
public:
    bool operator()(const Node &nodeA, const Node &nodeB) const{
        if(nodeA.name == nodeB.name){
            return true;
        }else{
            return false;
        }
    }
};

class Placeholders : public Node {
public:
    static Node getNewNode(){
        Node newNode = Node();
        newNode.isPlaceHolder = true;
        return newNode;
    }

    vector<vector<float>> compute(Node &node, vector<vector<float>> &input_vals) override {
        // input_vals[0]代表第一个数，input_vals[1]代表第二个数。
        // 这两个数可以是浮点数，也可以是矩阵。在这里是浮点数，并且存放在长度为1的vector中。
        vector<vector<float>> res;

        return res;
    }

    vector<Node> gradient(Node &node, vector<Node> &output_gradient) override{
        vector<Node> res;
        return res;
    }

    ~Placeholders() = default;
};



class AddOp : public Node {
public:
    AddOp() = default;

    static Node getNewNode(Node &nodeA, Node &nodeB) {
        Node newNode = Node();
//        newNode.op = this;
        newNode.input.push_back(nodeA);
        newNode.input.push_back(nodeB);
        newNode.name = nodeA.name + "+" + nodeB.name;
        newNode.isPlaceHolder = false;
        return newNode;
    }

    vector<vector<float>> compute(Node &node, vector<vector<float>> &input_vals) override {
        // input_vals[0]代表第一个数，input_vals[1]代表第二个数。
        // 这两个数可以是浮点数，也可以是矩阵。在这里是浮点数，并且存放在长度为1的vector中。
        vector<vector<float>> res;
        float value = input_vals[0][0] + input_vals[1][0];
        vector<float> t;
        t.push_back(value);
        res.push_back(t);
        return res;
    }

    vector<Node> gradient(Node &node, vector<Node> &output_gradient) override {
        vector<Node> res;
        res.push_back(output_gradient[0]);
        res.push_back(output_gradient[0]);
        return res;
    }

    ~AddOp()= default;
};

class AddByConstOp : public Node {
public:
    AddByConstOp() = default;

    static Node getNewNode(Node &nodeA, float &const_val) {
        Node newNode = Node();
//        newNode.op = this;
        newNode.input.push_back(nodeA);
        newNode.const_attr = const_val;
        newNode.isPlaceHolder = false;
        newNode.name = nodeA.name + "+" + std::to_string(const_val);
        return newNode;
    }

    vector<vector<float>> compute(Node &node, vector<vector<float>> &input_vals) override {
        // input_vals[0]代表第一个数，input_vals[1]代表第二个数。
        // 这两个数可以是浮点数，也可以是矩阵。在这里是浮点数，并且存放在长度为1的vector中。
        vector<vector<float>> res;
        float value = input_vals[0][0] + node.const_attr;
        vector<float> t;
        t.push_back(value);
        res.push_back(t);
        return res;
    }

    vector<Node> gradient(Node &node, vector<Node> &output_gradient) override {
        vector<Node> res;
        res.push_back(output_gradient[0]);
        return res;
    }

    ~AddByConstOp(){};
};

class ZerosLikeOp : public Node {
public:
    ZerosLikeOp() = default;

    static Node getNewNode(Node &nodeA) {
        Node newNode = Node();
        newNode.input.push_back(nodeA);
        newNode.name = "Zeroslike(" + nodeA.name + ")";
        return newNode;
    }

    vector<vector<float>> compute(Node &nodeA, vector<vector<float>> &input_vals){
        vector<float> zeros(input_vals[0].size());
        vector<vector<float>> res;
        res.push_back(zeros);
        return res;
    }

    vector<Node> gradient(Node &node, vector<Node> &output_gradient) override {
        vector<Node> res;
        Node newNode = Node();
        newNode.input.push_back(node.input[0]);
        newNode.name = "Zeroslike(" + node.input[0].name + ")";
        res.push_back(newNode);
    }

};

class OnesLikeOp : public Node {
public:
    OnesLikeOp() = default;

    static Node getNewNode(Node &nodeA) {
        Node newNode = Node();
        newNode.input.push_back(nodeA);
        newNode.name = "Oneslike(" + nodeA.name + ")";
        return newNode;
    }

    vector<vector<float>> compute(Node &node, vector<vector<float>> &input_vals){
        vector<float> ones(input_vals[0].size());
        vector<vector<float>> res;
        res.push_back(ones);
        return res;
    }

    vector<Node> gradient(Node &node, vector<Node> &output_gradient) override {
        vector<Node> res;
        Node newNode = Node();
        newNode.input.push_back(node.input[0]);
        newNode.name = "Zeroslike(" + node.input[0].name + ")";
        res.push_back(newNode);
    }

};

Placeholders placeholder_op = Placeholders();
AddOp add_op = AddOp();
AddByConstOp add_byconst_op = AddByConstOp();
OnesLikeOp ones_like_op = OnesLikeOp();


Node Variable(string var_name){
    Node placeholder_node = placeholder_op.getNewNode();
    placeholder_node.name = var_name;
    return placeholder_node;
}

Node Node::operator+(float &value){
    Node* nodeA = this;
    Node newNode = add_byconst_op.getNewNode(reinterpret_cast<Node &>(nodeA), value);
    return newNode;
}

Node Node::operator+(Node &nodeB){
    Node nodeA = *this;
    Node newNode = add_op.getNewNode(reinterpret_cast<Node &>(nodeA), nodeB);
    return newNode;
}

vector<vector<float>> Node::compute(Node &node, vector<vector<float>> &input_vals) {
    return {};
}

vector<Node> Node::gradient(Node &node, vector<Node> &output_gradient) {
    return {};
}


void topo_sort_dfs(Node &node, std::unordered_set<int> &visited, vector<Node> &topo_order){
    if(visited.count(node.hash_code)){
        return;
    }
    visited.emplace(node.hash_code);
    for(Node &in_node : node.input){
        topo_sort_dfs(in_node, visited, topo_order);
    }
    topo_order.push_back(node);
}

vector<Node> find_topo_sort(vector<Node> &node_list){
    std::unordered_set<int> visited;
    vector<Node> topo_order;
    for(Node &node : node_list){
        topo_sort_dfs(node, visited, topo_order);
    }
    return topo_order;
}

Node sum_node_list(vector<Node> node_list){
    Node res = node_list[0];
    int i = 1;
    int size = node_list.size();
    for(i = 1; i < size ; i++){
        res = res + node_list[i];
    }
    return res;
}

class Executor{
public:
    vector<Node> node_list;
    Executor(vector<Node> &eval_node_list){
        this->node_list = eval_node_list;
    }

    ~Executor(){};

    vector<vector<float>> run(map<Node, vector<float>>& feed_dic){
        vector<Node> topo_order = find_topo_sort(this->node_list);
        vector<vector<float>> input_vals; // 输入
        vector<vector<float>> output_vals; // 输出
        for(Node node : topo_order){
            if(node.isPlaceHolder){
                continue;
            }
            for(Node in_node : node.input){
                input_vals.push_back(feed_dic.at(in_node));
            }
            auto res = node.compute(node, input_vals);
            feed_dic[node] = res[0];
        }
        for(Node node : this->node_list){
            output_vals.push_back(feed_dic.at(node));
        }
        return output_vals;
    }
};

vector<Node> gradients(Node output_node, vector<Node> node_list){
    map<Node, vector<Node>> node_to_output_grads_list;
    vector<Node> ones;
    Node out_ones = OnesLikeOp::getNewNode(output_node);
    ones.push_back(out_ones);
    node_to_output_grads_list[output_node] = ones;

    map<Node, Node> node_to_output_grad;

    vector<Node> output_node_list;
    output_node_list.push_back(output_node);
    vector<Node> reverse_topo_sort = find_topo_sort(output_node_list);
    reverse(reverse_topo_sort.begin(), reverse_topo_sort.end());

    for(Node node : reverse_topo_sort){
        Node grad = sum_node_list(node_to_output_grads_list[node]);
        node_to_output_grad[node] = grad;
        vector<Node> grad_list;
        grad_list.push_back(grad);
        vector<Node> input_grads = node.gradient(node, grad_list);
        for(int i; i < node.input.size(); i++){
            node_to_output_grads_list[node.input[i]] = node_to_output_grads_list[node.input[i]];
            node_to_output_grads_list[node.input[i]].push_back(input_grads[i]);
        }
    }

    vector<Node> grad_node_list;
    for(Node node : node_list){
        grad_node_list.push_back(node_to_output_grad[node]);
    }
    return grad_node_list;
}

int main(){
    Node x1 = Variable("x1");
    Node x2 = Variable("x2");
    Node y = x1 + x2;

    vector<Node> input_nodes;
    input_nodes.push_back(x1);
    input_nodes.push_back(x2);

    vector<Node> grads = gradients(y, input_nodes);
    vector<Node> exe_list;
    exe_list.push_back(y);
    exe_list.push_back(x1);
    exe_list.push_back(x2);
    Executor executor = Executor(exe_list);

    vector<float> x1_val(1, 1);
    vector<float> x2_val(1, 2);
    map<Node, vector<float>> feed_dic;
    feed_dic[x1] = x1_val;
    feed_dic[x2] = x2_val;
    vector<vector<float>> res = executor.run(feed_dic);
    for(vector<float> node_res : res){
        cout<< node_res[0] << endl;
    }
}