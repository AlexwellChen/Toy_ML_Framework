#include <iostream>
#include "Model.h"
#include <Eigen/Dense>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace Eigen;

int main() {
    srand(42);

    // MLP: 784 -> Dense(128,relu) -> Dense(64,relu) -> Output(10)
    int BS = 64;
    Model model(BS, 5000, 0.005f, 0.95f, 1000);

    model.use_adam = true;
    model.adam_beta1 = 0.9f;
    model.adam_beta2 = 0.999f;
    model.adam_eps = 1e-8f;
    model.adam_wd = 0.0001f;

    model.log_interval = 100;
    model.csv_filename = "output/mnist_metrics.csv";

    model.Input(784, "input");
    model.Dense(128, "hidden1", "relu");
    model.Dense(64, "hidden2", "relu");
    model.Output(10, "output");

    model.compile();
    model.loadData("train.txt");
    model.run();

    model.save_weights("output/mnist_weights.bin");
    model.save_graph("output/mnist_graph.txt");

    return 0;
}
