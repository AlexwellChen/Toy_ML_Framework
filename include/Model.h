#include <iostream>
#include <map>
#include "../include/MatAutoDiff.h"
#include <Eigen/Dense>
#include <utility>

using namespace Eigen;
using namespace std;

class Layer {
public:
    int shape;
    Node input_node;
    Node layer;
    Node output;
    Node y_true;
    Node input;
    Node grad;
    int grad_node_hash;
    MatrixXd matrix_weight;
    MatrixXd y_true_val;

    Layer(int layer_shape, const string& layer_name, Layer &previous_layer, const string& activation);

    Layer(int input_shape, string input_name, int batch_size);

    Layer(int layer_shape, const string& layer_name, Layer &previous_layer, int batch_size);

    ~Layer() = default;
};

class Model {
public:
    vector<Layer> sequential;
    vector<Node> grads;
    Executor executor;
    vector<Node> exe_list;
    float lr = .1/batch_size;
    map<int, MatrixXd> feed_dic;
    vector<float> X_train;
    vector<float> y_train;
    int batch_size = 10;
    int training_iter = 100;

    Model() = default;

    Model(int batch_size_, int training_iter_);

    void Dense(int layer_shape, const string& layer_name, const string& activation);

    void Input(int input_shape, string input_name);

    void Output(int layer_shape, const string& layer_name);

    void compile();

    void loadData(vector<float> &X_train_, vector<float> y_train_);

    void run();

    ~Model() = default;

};