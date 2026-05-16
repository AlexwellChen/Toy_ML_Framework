#pragma once
#include <iostream>
#include <map>
#include <fstream>
#include "Node.h"
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

// ============================================================
// Layer
// ============================================================
enum LayerType { LAYER_INPUT, LAYER_DENSE, LAYER_OUTPUT };

class Layer {
public:
    int shape;
    LayerType layer_type;
    Node input_node;
    Node bias_node;
    Node layer;
    Node output;
    Node grad;
    int grad_node_hash;
    int bias_grad_node_hash;
    MatrixXd matrix_weight;
    MatrixXd bias_weight;
    Node input;
    Node y_true;
    MatrixXd y_true_val;

    Layer(int layer_shape, const string& layer_name, Layer &previous_layer, const string& activation);
    Layer(int input_shape, string input_name, int batch_size);
    Layer(int layer_shape, const string& layer_name, Layer &previous_layer, int batch_size);
    ~Layer() = default;
};

// ============================================================
// Training Metrics
// ============================================================
struct TrainingMetrics {
    int iteration;
    float loss;
    float train_acc, val_acc, test_acc;
    float train_auc, val_auc, test_auc;
    float learning_rate;
};

// ============================================================
// Data loading
// ============================================================
void load_data(vector<float> &X_train, vector<float> &y_train, string src);
void read_batch_data(MatrixXd &input_val, MatrixXd &y_true_val, vector<float> &X_train, vector<float> &y_train, int batch_size, int start_idx = -1);
void split_data(const vector<float> &X, const vector<float> &y, vector<float> &X_train, vector<float> &y_train, vector<float> &X_val, vector<float> &y_val, vector<float> &X_test, vector<float> &y_test, float train_ratio = 0.7f, float val_ratio = 0.15f);

// ============================================================
// Model
// ============================================================
class Model {
public:
    vector<Layer> sequential;
    vector<Node> grads;
    Executor executor, inference_executor;
    vector<Node> exe_list;
    float lr, initial_lr, lr_decay;
    int lr_decay_step;
    map<int, MatrixXd> feed_dic;
    vector<float> X_train, X_val, X_test;
    vector<float> y_train, y_val, y_test;
    int batch_size, training_iter, log_interval;
    vector<TrainingMetrics> metrics_history;
    string csv_filename;

    bool use_adam;
    float adam_beta1, adam_beta2, adam_eps, adam_wd;
    int adam_t;
    map<int, MatrixXd> adam_m, adam_v;

    Model() = default;
    Model(int batch_size_, int training_iter_, float lr_ = 0.1f, float lr_decay_ = 0.95f, int lr_decay_step_ = 100);

    void Dense(int layer_shape, const string& layer_name, const string& activation);
    void Input(int input_shape, string input_name);
    void Output(int layer_shape, const string& layer_name);

    void compile();
    void loadData(string data_path);
    void run();
    float evaluate(vector<float> &X, vector<float> &y);
    float calculate_auc_metric(vector<float> &X, vector<float> &y);
    void save_metrics_to_csv(const string& filename);
    void save_weights(const string& filename);
    void load_weights(const string& filename);
    void save_graph(const string& filename);

    ~Model() = default;
};
