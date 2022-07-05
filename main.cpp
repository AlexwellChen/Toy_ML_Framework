#include <iostream>
#include <map>
#include "include/Model.h"
#include <Eigen/Dense>
#include <utility>
#include <Eigen/Core>


using namespace std;
using namespace Eigen;

int main() {
    Eigen::initParallel();
    string training_data = "../train.txt";
    vector<float> X;
    vector<float> y;

    load_data(X, y, training_data);

    // Todo: training set and test set split; predict func; accuracy;


    Model model = Model(32, 100);

    model.Input(784, "input layer");
    model.Dense(128, "Hidden layer 1", "relu");
    model.Dense(64, "Hidden layer 2", "relu");
    model.Output(10, "Output layer");

    model.compile();
    model.loadData(X, y);
    model.run();
}

