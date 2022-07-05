#include <iostream>
#include <map>
#include "include/MatAutoDiff.h"
#include <Eigen/Dense>

#define BATCH_SIZE 10
#define ITER 100

using namespace Eigen;
using namespace std;

extern ReluOp relu_op;
extern SoftmaxOp softmax_op;

void NN_test(){
    map<int, int> node_to_gradient;
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
    cout << "------------------------------------" << endl;
    cout << "Building graph..." << endl;
    vector<Node> grads = gradients(y_predict, input_nodes);
    cout << "Building graph done..." << endl;


    node_to_gradient[W1.hash_code] = grads[1].hash_code;
    node_to_gradient[W2.hash_code] = grads[2].hash_code;
    node_to_gradient[W3.hash_code] = grads[3].hash_code;

    vector<Node> exe_list;
    exe_list.push_back(y_predict);
    exe_list.insert(exe_list.end(), grads.begin(), grads.end());
    Executor executor = Executor(exe_list);

    MatrixXd input_val(BATCH_SIZE, 784);
    MatrixXd W1_val = MatrixXd::Random(784,128);
    MatrixXd W2_val = MatrixXd::Random(128,64);
    MatrixXd W3_val = MatrixXd::Random(64,10);
    MatrixXd y_true_val(BATCH_SIZE, 10);

    float lr = .1/BATCH_SIZE;

    vector<float> X_train;
    vector<float> y_train;
    cout << "------------------------------------" << endl;
    cout << "Start loading data..." << endl;
    load_data(X_train, y_train);
    cout << "Finish loading data..." << endl;

    map<int, MatrixXd> feed_dic;
    feed_dic[W1.hash_code] = W1_val;
    feed_dic[W2.hash_code] = W2_val;
    feed_dic[W3.hash_code] = W3_val;

    int iter_;
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
        feed_dic[W1.hash_code] = feed_dic[W1.hash_code] - lr * feed_dic[node_to_gradient[W1.hash_code]];
        feed_dic[W2.hash_code] = feed_dic[W2.hash_code] - lr * feed_dic[node_to_gradient[W2.hash_code]];
        feed_dic[W3.hash_code] = feed_dic[W3.hash_code] - lr * feed_dic[node_to_gradient[W3.hash_code]];

        // Loss
        if(iter_ % 10 == 0){
            MatrixXd loss_m = feed_dic[y_predict.hash_code] - feed_dic[y_true.hash_code];
            cout << "Iteration: " << iter_ <<  ", Loss: " << loss_m.array().square().sum() / (BATCH_SIZE * 10) << endl;
            cout << "------------------------------------" << endl;
        }
    }
}

class Layer {
public:
    Layer(int num){
        this->node_num = num;
    }
    int node_num;

    ~Layer() = default;
};

class Model {
public:
    vector<Layer> 
};

int main() {
    map<int, int> node_to_gradient;
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
    cout << "------------------------------------" << endl;
    cout << "Building graph..." << endl;
    vector<Node> grads = gradients(y_predict, input_nodes);
    cout << "Building graph done..." << endl;


    node_to_gradient[W1.hash_code] = grads[1].hash_code;
    node_to_gradient[W2.hash_code] = grads[2].hash_code;
    node_to_gradient[W3.hash_code] = grads[3].hash_code;

    vector<Node> exe_list;
    exe_list.push_back(y_predict);
    exe_list.insert(exe_list.end(), grads.begin(), grads.end());
    Executor executor = Executor(exe_list);

    MatrixXd input_val(BATCH_SIZE, 784);
    MatrixXd W1_val = MatrixXd::Random(784,128);
    MatrixXd W2_val = MatrixXd::Random(128,64);
    MatrixXd W3_val = MatrixXd::Random(64,10);
    MatrixXd y_true_val(BATCH_SIZE, 10);

    float lr = .1/BATCH_SIZE;

    vector<float> X_train;
    vector<float> y_train;
    cout << "------------------------------------" << endl;
    cout << "Start loading data..." << endl;
    load_data(X_train, y_train);
    cout << "Finish loading data..." << endl;

    map<int, MatrixXd> feed_dic;
    feed_dic[W1.hash_code] = W1_val;
    feed_dic[W2.hash_code] = W2_val;
    feed_dic[W3.hash_code] = W3_val;

    int iter_;
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
        feed_dic[W1.hash_code] = feed_dic[W1.hash_code] - lr * feed_dic[node_to_gradient[W1.hash_code]];
        feed_dic[W2.hash_code] = feed_dic[W2.hash_code] - lr * feed_dic[node_to_gradient[W2.hash_code]];
        feed_dic[W3.hash_code] = feed_dic[W3.hash_code] - lr * feed_dic[node_to_gradient[W3.hash_code]];

        // Loss
        if(iter_ % 10 == 0){
            MatrixXd loss_m = feed_dic[y_predict.hash_code] - feed_dic[y_true.hash_code];
            cout << "Iteration: " << iter_ <<  ", Loss: " << loss_m.array().square().sum() / (BATCH_SIZE * 10) << endl;
            cout << "------------------------------------" << endl;
        }
    }
}

