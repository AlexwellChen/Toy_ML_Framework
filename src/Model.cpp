#include "Model.h"
#include "Relu.h"
#include "Softmax.h"
#include <Eigen/Dense>
#include <utility>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstdlib>

using namespace Eigen;
using namespace std;

extern ReluOp relu_op;
extern SoftmaxOp softmax_op;

// ============================================================
// Data loading helpers
// ============================================================
static vector<string> split_str(const string &s, char delim) {
    stringstream ss(s);
    string item;
    vector<string> tokens;
    while (getline(ss, item, delim)) tokens.push_back(item);
    return tokens;
}

static vector<float> operator/(const vector<float> &m2, const float m1) {
    const unsigned long n = m2.size();
    vector<float> product(n);
    for (unsigned i = 0; i != n; ++i) product[i] = m2[i] / m1;
    return product;
}

void load_data(vector<float> &X_train, vector<float> &y_train, string src) {
    string line;
    ifstream myfile(src);
    if (myfile.is_open()) {
        while (getline(myfile, line)) {
            auto line_v = split_str(line, '\t');
            int digit = strtof(line_v[0].c_str(), 0);
            for (unsigned i = 0; i < 10; ++i) {
                if (i == digit) y_train.emplace_back(1.);
                else y_train.emplace_back(0.);
            }
            int size = static_cast<int>(line_v.size());
            for (unsigned i = 1; i < size; ++i)
                X_train.emplace_back(strtof(line_v[i].c_str(), 0));
        }
        X_train = X_train / 255.0;
        myfile.close();
    }
}

void read_batch_data(MatrixXd &input_val, MatrixXd &y_true_val, vector<float> &X_train, vector<float> &y_train, int batch_size, int start_idx) {
    int num_samples = X_train.size() / 784;
    int actual_start = start_idx;
    if (start_idx == -1) actual_start = rand() % (num_samples - batch_size + 1);
    for (int i = 0; i < batch_size; ++i) {
        int sample_idx = actual_start + i;
        if (sample_idx >= num_samples) sample_idx -= num_samples;
        for (int j = 0; j < 784; ++j) input_val(i, j) = X_train[sample_idx * 784 + j];
        for (int j = 0; j < 10; ++j) y_true_val(i, j) = y_train[sample_idx * 10 + j];
    }
}

void split_data(const vector<float> &X, const vector<float> &y, vector<float> &X_train, vector<float> &y_train, vector<float> &X_val, vector<float> &y_val, vector<float> &X_test, vector<float> &y_test, float train_ratio, float val_ratio) {
    int num_samples = X.size() / 784;
    int train_end = (int)(num_samples * train_ratio);
    int val_end = train_end + (int)(num_samples * val_ratio);
    vector<int> indices(num_samples);
    for (int i = 0; i < num_samples; ++i) indices[i] = i;
    srand(42);
    for (int i = num_samples - 1; i > 0; --i) { int j = rand() % (i + 1); swap(indices[i], indices[j]); }
    for (int i = 0; i < num_samples; ++i) {
        int idx = indices[i];
        if (i < train_end) {
            for (int j = 0; j < 784; ++j) X_train.push_back(X[idx * 784 + j]);
            for (int j = 0; j < 10; ++j) y_train.push_back(y[idx * 10 + j]);
        } else if (i < val_end) {
            for (int j = 0; j < 784; ++j) X_val.push_back(X[idx * 784 + j]);
            for (int j = 0; j < 10; ++j) y_val.push_back(y[idx * 10 + j]);
        } else {
            for (int j = 0; j < 784; ++j) X_test.push_back(X[idx * 784 + j]);
            for (int j = 0; j < 10; ++j) y_test.push_back(y[idx * 10 + j]);
        }
    }
    cout << "Data split complete: " << endl;
    cout << "  Train samples: " << X_train.size() / 784 << endl;
    cout << "  Val samples:   " << X_val.size() / 784 << endl;
    cout << "  Test samples:  " << X_test.size() / 784 << endl;
}

// ============================================================
// Layer constructors
// ============================================================
Layer::Layer(int layer_shape, const string& layer_name, Layer &previous_layer, const string& activation){
    this->layer_type = LAYER_DENSE;
    this->shape = layer_shape;
    this->input_node = Variable(layer_name + "_W");
    this->bias_node = Variable(layer_name + "_b");
    Node matmul_out = previous_layer.output * input_node;
    this->layer = matmul_out + bias_node;
    this->output = (activation == "relu") ? relu_op.getNewNode(this->layer) : this->layer;
    float he_scale = sqrt(2.0f / previous_layer.shape);
    matrix_weight = MatrixXd::Random(previous_layer.shape, layer_shape) * he_scale;
    bias_weight = MatrixXd::Zero(1, layer_shape);
    bias_grad_node_hash = -1;
}

Layer::Layer(int input_shape, string input_name, int batch_size){
    this->layer_type = LAYER_INPUT;
    this->shape = input_shape;
    input = Variable(std::move(input_name));
    this->input_node = this->input;
    this->layer = this->input;
    this->output = this->input;
    matrix_weight = MatrixXd::Random(batch_size, input_shape);
    bias_grad_node_hash = -1;
}

Layer::Layer(int layer_shape, const string& layer_name, Layer &previous_layer, int batch_size){
    this->layer_type = LAYER_OUTPUT;
    this->shape = layer_shape;
    this->y_true = Variable("Ground Truth");
    this->input_node = Variable(layer_name + "_W");
    this->bias_node = Variable(layer_name + "_b");
    Node matmul_out = previous_layer.output * input_node;
    this->layer = matmul_out + bias_node;
    this->output = softmax_op.getNewNode(this->layer, y_true);
    float he_scale = sqrt(2.0f / previous_layer.shape);
    matrix_weight = MatrixXd::Random(previous_layer.shape, layer_shape) * he_scale;
    bias_weight = MatrixXd::Zero(1, layer_shape);
    bias_grad_node_hash = -1;
    y_true_val = MatrixXd::Random(batch_size, layer_shape);
}

// ============================================================
// Model methods
// ============================================================
Model::Model(int batch_size_, int training_iter_, float lr_, float lr_decay_, int lr_decay_step_) :
    batch_size(batch_size_), training_iter(training_iter_), lr(lr_), initial_lr(lr_),
    lr_decay(lr_decay_), lr_decay_step(lr_decay_step_), log_interval(50),
    csv_filename("training_metrics.csv"), use_adam(false),
    adam_beta1(0.9f), adam_beta2(0.999f), adam_eps(1e-8f), adam_wd(0.01f), adam_t(0) {}

void Model::Dense(int layer_shape, const string& layer_name, const string& activation){
    sequential.emplace_back(Layer(layer_shape, layer_name, sequential.back(), activation));
}

void Model::Input(int input_shape, string input_name){
    if(!sequential.empty()){ sequential.clear(); cout << "Model has been cleared!" << endl; }
    sequential.emplace_back(Layer(input_shape, std::move(input_name), batch_size));
}

void Model::Output(int layer_shape, const string& layer_name){
    sequential.emplace_back(Layer(layer_shape, layer_name, sequential.back(), batch_size));
}

void Model::compile(){
    cout << "------------------------------------" << endl;
    cout << "Building graph..." << endl;

    vector<int> node_to_layer;
    vector<bool> node_is_bias;
    vector<Node> input_nodes;
    for(int i = 0; i < (int)sequential.size(); i++){
        Layer &layer = sequential[i];
        node_to_layer.push_back(i);
        node_is_bias.push_back(false);
        input_nodes.emplace_back(layer.input_node);
        if (layer.matrix_weight.size() > 0) feed_dic[layer.input_node.hash_code] = layer.matrix_weight;
        if ((layer.layer_type == LAYER_DENSE || layer.layer_type == LAYER_OUTPUT) && layer.bias_weight.size() > 0) {
            node_to_layer.push_back(i);
            node_is_bias.push_back(true);
            input_nodes.emplace_back(layer.bias_node);
            feed_dic[layer.bias_node.hash_code] = layer.bias_weight;
        }
    }
    input_nodes.emplace_back(sequential.back().y_true);
    grads = gradients(sequential.back().output, input_nodes);

    for(int i = 0; i < (int)sequential.size(); i++){
        sequential[i].grad_node_hash = -1;
        sequential[i].bias_grad_node_hash = -1;
    }
    for(int gi = 0; gi < (int)node_to_layer.size(); gi++){
        int li = node_to_layer[gi];
        if (node_is_bias[gi]) sequential[li].bias_grad_node_hash = grads[gi].hash_code;
        else sequential[li].grad_node_hash = grads[gi].hash_code;
    }

    cout << "Build graph done!" << endl;
    cout << "------------------------------------" << endl;
    cout << "Building Executor..." << endl;
    exe_list.push_back(sequential.back().output);
    exe_list.insert(exe_list.end(), grads.begin(), grads.end());
    executor = Executor(exe_list);
    inference_executor = Executor({sequential.back().output});
    cout << "Build Executor done!" << endl;
}

void Model::loadData(string data_path) {
    vector<float> X_all, y_all;
    load_data(X_all, y_all, data_path);
    split_data(X_all, y_all, X_train, y_train, X_val, y_val, X_test, y_test);
}

float Model::evaluate(vector<float> &X, vector<float> &y) {
    int num_samples = X.size() / 784, correct = 0;
    int eval_bs = min(batch_size, 100);
    MatrixXd input_val(eval_bs, 784), y_true_val(eval_bs, 10);
    for (int i = 0; i < num_samples; i += eval_bs) {
        int actual = min(eval_bs, num_samples - i);
        read_batch_data(input_val, y_true_val, X, y, actual, i);
        feed_dic[sequential.front().input_node.hash_code] = input_val.topRows(actual);
        feed_dic[sequential.back().y_true.hash_code] = y_true_val.topRows(actual);
        inference_executor.run(feed_dic);
        MatrixXd y_pred = feed_dic[sequential.back().output.hash_code];
        for (int j = 0; j < actual; ++j) {
            int pl = 0, tl = 0;
            float mp = y_pred(j,0), mt = y_true_val(j,0);
            for (int k = 1; k < 10; ++k) {
                if (y_pred(j,k) > mp) { mp = y_pred(j,k); pl = k; }
                if (y_true_val(j,k) > mt) { mt = y_true_val(j,k); tl = k; }
            }
            if (pl == tl) correct++;
        }
    }
    return (float)correct / num_samples;
}

float Model::calculate_auc_metric(vector<float> &X, vector<float> &y) {
    if (X.empty()) return 0.0f;
    int num_samples = X.size() / 784;
    vector<pair<float, int>> all_preds;
    int eval_bs = min(batch_size, 100);
    MatrixXd input_val(eval_bs, 784), y_true_val(eval_bs, 10);
    for (int i = 0; i < num_samples; i += eval_bs) {
        int actual = min(eval_bs, num_samples - i);
        read_batch_data(input_val, y_true_val, X, y, actual, i);
        feed_dic[sequential.front().input_node.hash_code] = input_val.topRows(actual);
        feed_dic[sequential.back().y_true.hash_code] = y_true_val.topRows(actual);
        inference_executor.run(feed_dic);
        MatrixXd y_pred = feed_dic[sequential.back().output.hash_code];
        for (int j = 0; j < actual; ++j) {
            int tl = 0; float mt = y_true_val(j,0);
            for (int k = 1; k < 10; ++k) if (y_true_val(j,k) > mt) { mt = y_true_val(j,k); tl = k; }
            all_preds.emplace_back(y_pred(j, tl), tl);
        }
    }
    float total_auc = 0.0f; int valid_classes = 0;
    for (int c = 0; c < 10; ++c) {
        vector<pair<float, bool>> binary_preds;
        for (auto& p : all_preds) binary_preds.emplace_back(p.first, p.second == c);
        sort(binary_preds.begin(), binary_preds.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        long long pos = 0, neg = 0;
        for (auto& p : binary_preds) { if (p.second) pos++; else neg++; }
        if (pos > 0 && neg > 0) {
            long long auc_num = 0, correct = 0;
            for (auto& p : binary_preds) { if (p.second) correct++; else auc_num += correct; }
            total_auc += (float)auc_num / (pos * neg);
            valid_classes++;
        }
    }
    return valid_classes > 0 ? total_auc / valid_classes : 0.5f;
}

void Model::save_metrics_to_csv(const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) { cerr << "Could not open: " << filename << endl; return; }
    file << "iteration,loss,train_acc,val_acc,test_acc,train_auc,val_auc,test_auc,learning_rate\n";
    for (const auto& m : metrics_history)
        file << m.iteration << "," << m.loss << "," << m.train_acc << "," << m.val_acc << ","
             << m.test_acc << "," << m.train_auc << "," << m.val_auc << "," << m.test_auc << ","
             << m.learning_rate << "\n";
    file.close();
    cout << "Metrics saved to " << filename << endl;
}

void Model::run(){
    cout << "------------------------------------" << endl;
    cout << "START TRAINING" << endl;
    cout << "------------------------------------" << endl;
    metrics_history.clear();
    float best_val_acc = -1.0f;
    map<int, MatrixXd> best_weights;

    for(int iter_ = 0; iter_ < training_iter; iter_++) {
        if (iter_ > 0 && iter_ % lr_decay_step == 0) lr *= lr_decay;

        MatrixXd flat_input(batch_size, 784);
        read_batch_data(flat_input, sequential.back().y_true_val, X_train, y_train, batch_size, -1);
        sequential.front().matrix_weight = flat_input;
        feed_dic[sequential.front().input_node.hash_code] = flat_input;
        feed_dic[sequential.back().y_true.hash_code] = sequential.back().y_true_val;

        executor.run(feed_dic);

        if (use_adam) {
            adam_t++;
            float bc1 = 1.0f - pow(adam_beta1, adam_t);
            float bc2 = 1.0f - pow(adam_beta2, adam_t);
            for (int i = 1; i < (int)sequential.size(); i++) {
                auto& layer = sequential[i];
                int wh = layer.input_node.hash_code;
                MatrixXd gw = feed_dic[layer.grad_node_hash] / (double)batch_size;
                if (adam_m.find(wh) == adam_m.end()) {
                    adam_m[wh] = MatrixXd::Zero(gw.rows(), gw.cols());
                    adam_v[wh] = MatrixXd::Zero(gw.rows(), gw.cols());
                }
                adam_m[wh] = (adam_beta1 * adam_m[wh].array() + (1.0f - adam_beta1) * gw.array()).matrix();
                adam_v[wh] = (adam_beta2 * adam_v[wh].array() + (1.0f - adam_beta2) * gw.array().square()).matrix();
                feed_dic[wh] = ((1.0f - lr * adam_wd) * feed_dic[wh].array()
                    - lr * (adam_m[wh] / bc1).array() / ((adam_v[wh] / bc2).array().sqrt() + adam_eps)).matrix();

                if (layer.bias_grad_node_hash >= 0) {
                    int bh = layer.bias_node.hash_code;
                    MatrixXd gb_raw = feed_dic[layer.bias_grad_node_hash] / (double)batch_size;
                    MatrixXd& b = feed_dic[bh];
                    MatrixXd gb = gb_raw;
                    if (gb.rows() > b.rows()) gb = gb.colwise().sum().eval();
                    if (adam_m.find(bh) == adam_m.end()) {
                        adam_m[bh] = MatrixXd::Zero(b.rows(), b.cols());
                        adam_v[bh] = MatrixXd::Zero(b.rows(), b.cols());
                    }
                    adam_m[bh] = (adam_beta1 * adam_m[bh].array() + (1.0f - adam_beta1) * gb.array()).matrix();
                    adam_v[bh] = (adam_beta2 * adam_v[bh].array() + (1.0f - adam_beta2) * gb.array().square()).matrix();
                    b = ((1.0f - lr * adam_wd) * b.array()
                        - lr * (adam_m[bh] / bc1).array() / ((adam_v[bh] / bc2).array().sqrt() + adam_eps)).matrix();
                }
            }
        } else {
            for (int i = 1; i < (int)sequential.size(); i++) {
                feed_dic[sequential[i].input_node.hash_code] -= (lr / batch_size) * feed_dic[sequential[i].grad_node_hash];
                if (sequential[i].bias_grad_node_hash >= 0) {
                    MatrixXd gb = feed_dic[sequential[i].bias_grad_node_hash] / (double)batch_size;
                    MatrixXd& b = feed_dic[sequential[i].bias_node.hash_code];
                    if (gb.rows() > b.rows()) gb = gb.colwise().sum().eval();
                    b -= lr * gb;
                }
            }
        }

        if (iter_ % log_interval == 0 || iter_ == training_iter - 1) {
            MatrixXd loss_m = feed_dic[sequential.back().output.hash_code] - feed_dic[sequential.back().y_true.hash_code];
            float loss = loss_m.array().square().sum() / (batch_size * 10);
            float train_acc = evaluate(X_train, y_train);
            float val_acc = X_val.empty() ? 0.0f : evaluate(X_val, y_val);
            float test_acc = X_test.empty() ? 0.0f : evaluate(X_test, y_test);

            TrainingMetrics m;
            m.iteration = iter_; m.loss = loss;
            m.train_acc = train_acc; m.val_acc = val_acc; m.test_acc = test_acc;
            m.train_auc = calculate_auc_metric(X_train, y_train);
            m.val_auc = X_val.empty() ? 0.0f : calculate_auc_metric(X_val, y_val);
            m.test_auc = X_test.empty() ? 0.0f : calculate_auc_metric(X_test, y_test);
            m.learning_rate = lr;
            metrics_history.push_back(m);

            if (val_acc > best_val_acc) {
                best_val_acc = val_acc;
                for (int i = 1; i < (int)sequential.size(); i++) {
                    best_weights[sequential[i].input_node.hash_code] = feed_dic[sequential[i].input_node.hash_code];
                    if (sequential[i].bias_grad_node_hash >= 0)
                        best_weights[sequential[i].bias_node.hash_code] = feed_dic[sequential[i].bias_node.hash_code];
                }
                cout << "  [Best model saved, val_acc=" << best_val_acc << "]" << endl;
            }
            cout << "Iteration: " << iter_ << ", Loss: " << loss << ", Train Acc: " << train_acc
                 << ", Val Acc: " << val_acc << ", Test Acc: " << test_acc
                 << ", LR: " << lr << endl;
            cout << "------------------------------------" << endl;
        }
    }

    save_metrics_to_csv(csv_filename);

    if (!best_weights.empty()) {
        for (int i = 1; i < (int)sequential.size(); i++) {
            feed_dic[sequential[i].input_node.hash_code] = best_weights[sequential[i].input_node.hash_code];
            if (sequential[i].bias_grad_node_hash >= 0 && best_weights.count(sequential[i].bias_node.hash_code))
                feed_dic[sequential[i].bias_node.hash_code] = best_weights[sequential[i].bias_node.hash_code];
        }
        cout << "Restored best model (val_acc=" << best_val_acc << ")" << endl;
    }
    if (!X_test.empty()) {
        cout << "Final Test Accuracy: " << evaluate(X_test, y_test) << endl;
        cout << "------------------------------------" << endl;
    }
}

void Model::save_weights(const string& filename) {
    ofstream file(filename, ios::binary);
    if (!file.is_open()) { cerr << "Could not open: " << filename << endl; return; }
    int num_layers = sequential.size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(int));
    for (auto& layer : sequential) {
        MatrixXd& w = feed_dic[layer.input_node.hash_code];
        int rows = w.rows(), cols = w.cols();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        file.write(reinterpret_cast<const char*>(w.data()), rows * cols * sizeof(double));
        int has_bias = (layer.bias_grad_node_hash >= 0) ? 1 : 0;
        file.write(reinterpret_cast<const char*>(&has_bias), sizeof(int));
        if (has_bias) {
            MatrixXd& b = feed_dic[layer.bias_node.hash_code];
            int brows = b.rows(), bcols = b.cols();
            file.write(reinterpret_cast<const char*>(&brows), sizeof(int));
            file.write(reinterpret_cast<const char*>(&bcols), sizeof(int));
            file.write(reinterpret_cast<const char*>(b.data()), brows * bcols * sizeof(double));
        }
    }
    file.close();
    cout << "Weights saved to " << filename << endl;
}

void Model::load_weights(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) { cerr << "Could not open: " << filename << endl; return; }
    int num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(int));
    int layer_idx = 0;
    for (auto& layer : sequential) {
        if (layer_idx >= num_layers) break;
        int rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        MatrixXd w(rows, cols);
        file.read(reinterpret_cast<char*>(w.data()), rows * cols * sizeof(double));
        feed_dic[layer.input_node.hash_code] = w;
        layer.matrix_weight = w;
        int has_bias;
        file.read(reinterpret_cast<char*>(&has_bias), sizeof(int));
        if (has_bias) {
            int brows, bcols;
            file.read(reinterpret_cast<char*>(&brows), sizeof(int));
            file.read(reinterpret_cast<char*>(&bcols), sizeof(int));
            MatrixXd b(brows, bcols);
            file.read(reinterpret_cast<char*>(b.data()), brows * bcols * sizeof(double));
            feed_dic[layer.bias_node.hash_code] = b;
            layer.bias_weight = b;
        }
        layer_idx++;
    }
    file.close();
    cout << "Weights loaded from " << filename << endl;
}

void Model::save_graph(const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) { cerr << "Could not open: " << filename << endl; return; }
    file << batch_size << " " << training_iter << " " << initial_lr << " " << lr_decay << " " << lr_decay_step << "\n";
    for (auto& layer : sequential) {
        switch (layer.layer_type) {
        case LAYER_INPUT:  file << "INPUT " << layer.shape << " " << layer.input.name << "\n"; break;
        case LAYER_DENSE:  file << "DENSE " << layer.shape << " " << layer.input_node.name << " "
                                << (layer.output.name.find("Relu") != string::npos ? "relu" : "linear") << "\n"; break;
        case LAYER_OUTPUT: file << "OUTPUT " << layer.shape << " " << layer.input_node.name << "\n"; break;
        }
    }
    file.close();
    cout << "Graph saved to " << filename << endl;
}
