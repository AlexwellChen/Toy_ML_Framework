#include <iostream>
#include "../include/Model.h"
#include <Eigen/Dense>
#include <utility>

using namespace Eigen;
using namespace std;

extern ReluOp relu_op;
extern SoftmaxOp softmax_op;


Layer::Layer(int layer_shape, const string& layer_name, Layer &previous_layer, const string& activation){
    // Dense layer
    this->shape = layer_shape;
    this->input_node = Variable(std::move(layer_name + " weights"));
    this->layer = previous_layer.output * input_node;
    if(activation == "relu"){
        this->output = relu_op.getNewNode(this->layer);
    }else{
        this->output = this->layer;
    }
    matrix_weight = MatrixXd::Random(previous_layer.shape,layer_shape);
}

Layer::Layer(int input_shape, string input_name, int batch_size){
    // Input layer
    this->shape = input_shape;
    input = Variable(std::move(input_name));
    this->input_node = this->input;
    this->layer = this->input;
    this->output = this->input;
    matrix_weight = MatrixXd::Random(batch_size, input_shape);
}

Layer::Layer(int layer_shape, const string& layer_name, Layer &previous_layer, int batch_size){
    // Output layer
    this->shape = layer_shape;
    this->y_true = Variable("Ground Truth");
    this->input_node = Variable(std::move(layer_name + " weights"));
    this->layer = previous_layer.output * input_node;
    this->output = softmax_op.getNewNode(this->layer, y_true);
    matrix_weight = MatrixXd::Random(previous_layer.shape,layer_shape);
    y_true_val = MatrixXd::Random(batch_size, layer_shape);
}


Model::Model(int batch_size_, int training_iter_){
    this->lr = .1/batch_size_;
    this->batch_size = batch_size_;
    this->training_iter = training_iter_;
}

void Model::Dense(int layer_shape, const string& layer_name, const string& activation){
    if(sequential.empty()){
        cout << "Need input layer!" << endl;
        return;
    }
    Layer new_Layer = Layer(layer_shape, layer_name, sequential.back(), activation);
    sequential.emplace_back(new_Layer);
}

void Model::Input(int input_shape, string input_name){
    if(!sequential.empty()){
        cout << "Model not empty!" << endl;
        sequential.clear();
        cout << "Model has been cleared!" << endl;
    }
    sequential.emplace_back(Layer(input_shape, std::move(input_name), batch_size));
}

void Model::Output(int layer_shape, const string& layer_name){
    Layer new_Layer = Layer(layer_shape, layer_name, sequential.back(), batch_size);
    sequential.emplace_back(new_Layer);
}

void Model::compile(){
    cout << "------------------------------------" << endl;
    cout << "Building graph..." << endl;
    vector<Node> input_nodes;
    for(Layer &layer : sequential){
        // 将模型中每一层的节点送入
        input_nodes.emplace_back(layer.input_node);
        feed_dic[layer.input_node.hash_code] = layer.matrix_weight;
    }
    // 将输出层的真实值节点送入
    input_nodes.emplace_back(sequential.back().y_true);
    grads = gradients(sequential.back().output, input_nodes);
    for(int i = 0; i < grads.size(); i++){
        sequential[i].grad_node_hash = grads[i].hash_code;
    }
    cout << "Build graph done..." << endl;
    cout << "------------------------------------" << endl;
    cout << "Building Executor..." << endl;
    exe_list.push_back(sequential.back().output);
    exe_list.insert(exe_list.end(), grads.begin(), grads.end());
    executor = Executor(exe_list);
    cout << "Build Executor done..." << endl;
}

void Model::loadData(vector<float> &X_train_, vector<float> y_train_){
    this->X_train = X_train_;
    this->y_train = y_train_;
}

void Model::run(){
    int iter_;
    cout << "------------------------------------" << endl;
    cout << "START TRAINING" << endl;
    cout << "------------------------------------" << endl;
    for(iter_ = 0; iter_ < training_iter; iter_++) {
        // Read training data
        read_batch_data(sequential.front().matrix_weight, sequential.back().y_true_val, X_train, y_train, batch_size);
        feed_dic[sequential.front().input_node.hash_code] = sequential.front().matrix_weight;
        feed_dic[sequential.back().y_true.hash_code] = sequential.back().y_true_val;

        // Train
        vector<MatrixXd> res = executor.run(feed_dic);

        // Weight update
        for (int i = 1; i < sequential.size() - 1; i++) {
            feed_dic[sequential[i].input_node.hash_code] -= lr * feed_dic[sequential[i].grad_node_hash];
        }

        // Loss
        if (iter_ % 1 == 0) {
            MatrixXd loss_m = feed_dic[sequential.back().output.hash_code] - feed_dic[sequential.back().y_true.hash_code];
            cout << "Iteration: " << iter_ << ", Loss: " << loss_m.array().square().sum() / (batch_size * 10) << "\r" << flush;
        }
    }
}

