#include <iostream>
#include <string>
#include <unordered_set>
#include <map>
#include "include/deep_core.h"
#include "include/Op.h"

using namespace std;

class Node {
public:
    Node(){
        this->const_attr = 0.0;
    }
    vector<Node> input;
    Op *op;
    float const_attr;
    string name;
    int hash_code;
    bool isPlaceHolder;

    virtual Node operator+(Node &nodeB);

    virtual Node operator*(Node &nodeB);

    bool operator<(const Node& node) const{
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

vector<vector<float>> Op::compute(Node &node, vector<vector<float>> &input_vals) {
    return vector<vector<float>>();
}

vector<Node> Op::gradient(Node &node, Node &output_gradient) {
    return vector<Node>();
}

class Placeholders : public Op {
public:
    Node getNewNode(){
        Node newNode = Node();
        newNode.isPlaceHolder = true;
        newNode.op = this;
        return newNode;
    }

    vector<vector<float>> compute(Node &node, vector<vector<float>> &input_vals) override {
        // input_vals[0]代表第一个数，input_vals[1]代表第二个数。
        // 这两个数可以是浮点数，也可以是矩阵。在这里是浮点数，并且存放在长度为1的vector中。
        vector<vector<float>> res;
        return res;
    }

    vector<Node> gradient(Node &node, Node &output_gradient) override{
        vector<Node> res;
        return res;
    }

    ~Placeholders() = default;
};



class AddOp : public Op {
public:
    AddOp() = default;

    Node getNewNode(Node &nodeA, Node &nodeB) {
        Node newNode = Node();
        newNode.op = this;
        newNode.input.push_back(nodeA);
        newNode.input.push_back(nodeB);
        newNode.name = nodeA.name + " + " + nodeB.name;
        newNode.isPlaceHolder = false;
        return newNode;
    }

    vector<vector<float>> compute(Node &node, vector<vector<float>> &input_vals) override {
        // input_vals[0]代表第一个数，input_vals[1]代表第二个数。
        // 这两个数可以是浮点数，也可以是矩阵。在这里是浮点数，并且存放在长度为1的vector中。
        vector<vector<float>> res;
        int size = input_vals[0].size();
        vector<float> t;
        for(int i = 0; i< size; i++){
            t.push_back(input_vals[0][i] + input_vals[1][i]);
        }
        res.push_back(t);
        return res;
    }

    vector<Node> gradient(Node &node, Node &output_gradient) override {
        vector<Node> res;
        res.push_back(output_gradient);
        res.push_back(output_gradient);
        return res;
    }

    ~AddOp()= default;
};

class AddByConstOp : public Op {
public:
    AddByConstOp() = default;

    Node getNewNode(Node &nodeA, float &const_val) {
        Node newNode = Node();
        newNode.op = this;
        newNode.input.push_back(nodeA);
        newNode.const_attr = const_val;
        newNode.isPlaceHolder = false;
        newNode.name = nodeA.name + " + " + std::to_string(const_val);
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

    vector<Node> gradient(Node &node, Node &output_gradient) override {
        vector<Node> res;
        res.push_back(output_gradient);
        return res;
    }

    ~AddByConstOp(){};
};

class MulOp : public Op {
public:
    MulOp() = default;
    Node getNewNode(Node &nodeA, Node &nodeB) {
        Node newNode = Node();
        newNode.op = this;
        newNode.input.push_back(nodeA);
        newNode.input.push_back(nodeB);
        newNode.isPlaceHolder = false;
        newNode.name = nodeA.name + " * " + nodeB.name;
        return newNode;
    }

    vector<vector<float>> compute(Node &node, vector<vector<float>> &input_vals) override {
        // input_vals[0]代表第一个数，input_vals[1]代表第二个数。
        // 这两个数可以是浮点数，也可以是矩阵。在这里是浮点数，并且存放在长度为1的vector中。
        vector<vector<float>> res;
        int size = input_vals[0].size();
        vector<float> t;
        for(int i = 0; i < size; i++){
            t.push_back(input_vals[0][i] * input_vals[1][i]);
        }
        res.push_back(t);
        return res;
    }

    vector<Node> gradient(Node &node, Node &output_gradient) override {
        vector<Node> res;
        Node out1 = MulOp::getNewNode(node.input[1], output_gradient);
        Node out2 = MulOp::getNewNode(node.input[0], output_gradient);
        res.push_back(out1);
        res.push_back(out2);
        return res;
    }

};

class ZerosLikeOp : public Op {
public:
    ZerosLikeOp() = default;

    Node getNewNode(Node &nodeA) {
        Node newNode = Node();
        newNode.op = this;
        newNode.input.push_back(nodeA);
        newNode.name = "Zeroslike(" + nodeA.name + ")";
        newNode.isPlaceHolder = false;
        return newNode;
    }

    vector<vector<float>> compute(Node &nodeA, vector<vector<float>> &input_vals){
        vector<float> zeros(input_vals[0].size(), 0);
        vector<vector<float>> res;
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
        return newNode;
    }

    vector<vector<float>> compute(Node &node, vector<vector<float>> &input_vals) override {
        vector<float> ones(input_vals[0].size(), 1);
        vector<vector<float>> res;
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

};

Placeholders placeholder_op = Placeholders();
AddOp add_op = AddOp();
AddByConstOp add_byconst_op = AddByConstOp();
OnesLikeOp ones_like_op = OnesLikeOp();
MulOp mul_op = MulOp();


Node Variable(string var_name){
    Node placeholder_node = placeholder_op.getNewNode();
    placeholder_node.name = var_name;
    int hash = 0;
    int i = 0;
    for (hash = var_name.length(), i = 0; i < var_name.length(); i++)
        hash += var_name[i];
    placeholder_node.hash_code = (hash % 31);
    return placeholder_node;
}

//Node Node::operator+(float &value){
//    Node* nodeA = this;
//    Node newNode = add_byconst_op.getNewNode(reinterpret_cast<Node &>(nodeA), value);
//    return newNode;
//}

Node Node::operator+(Node &nodeB){
    Node nodeA = *this;
    Node newNode = add_op.getNewNode(nodeA, nodeB);
    return newNode;
}

Node Node::operator*(Node &nodeB){
    Node nodeA = *this;
    Node newNode = mul_op.getNewNode(nodeA, nodeB);
    return newNode;
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

    ~Executor() = default;

    vector<vector<float>> run(map<int, vector<float>>& feed_dic){
        vector<Node> topo_order = find_topo_sort(this->node_list);
        vector<vector<float>> input_vals; // 输入
        vector<vector<float>> output_vals; // 输出
        for(Node node : topo_order){
            if(node.isPlaceHolder){
                continue;
            }
            input_vals.clear();
            for(Node in_node : node.input){
                input_vals.push_back(feed_dic[in_node.hash_code]);
            }
            auto res = node.op->compute(node, input_vals);
            feed_dic[node.hash_code] = res[0];
        }
        for(Node node : this->node_list){
            output_vals.push_back(feed_dic[node.hash_code]);
        }
        return output_vals;
    }
};

vector<Node> gradients(Node output_node, vector<Node> node_list){
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

    for(Node node : reverse_topo_sort){
        Node grad = sum_node_list(node_to_output_grads_list[node.hash_code]);
        node_to_output_grad[node.hash_code] = grad;
        vector<Node> input_grads = node.op->gradient(node, grad); // 计算当前节点前驱的梯度
        for(int i = 0; i < node.input.size(); i++){
            node_to_output_grads_list[node.input[i].hash_code] = node_to_output_grads_list[node.input[i].hash_code];
            node_to_output_grads_list[node.input[i].hash_code].push_back(input_grads[i]);
        }
    }

    vector<Node> grad_node_list;
    for(Node node : node_list){
        grad_node_list.push_back(node_to_output_grad[node.hash_code]);
    }
    return grad_node_list;
}

int main(){
    Node x1 = Variable("x1");
    Node x2 = Variable("x2");
    Node x3 = Variable("x3");
    Node y = x1 * x2 + x3;

    vector<Node> input_nodes;
    input_nodes.push_back(x1);
    input_nodes.push_back(x2);
    input_nodes.push_back(x3);
    vector<Node> grads = gradients(y, input_nodes);

    vector<Node> exe_list;
    exe_list.push_back(y);
    exe_list.push_back(grads[0]);
    exe_list.push_back(grads[1]);
    exe_list.push_back(grads[2]);
    Executor executor = Executor(exe_list);

    vector<float> x1_val(3, 1);
    vector<float> x2_val(3, 2);
    vector<float> x3_val(3, 4);
    map<int, vector<float>> feed_dic;
    feed_dic[x1.hash_code] = x1_val;
    feed_dic[x2.hash_code] = x2_val;
    feed_dic[x3.hash_code] = x3_val;
    vector<vector<float>> res = executor.run(feed_dic);
    for(int i = 0; i < exe_list.size(); i++){
        cout<<"Node: "<< exe_list[i].name << " value is: "<< res[i][0]<<endl;
    }
}