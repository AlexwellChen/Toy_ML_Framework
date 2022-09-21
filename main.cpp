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

    cout << "------------------------------------" << endl;
    cout << "Loading data..." << endl;
    load_data(X, y, training_data);
    cout << "Finish loading data..." << endl;

    Model model = Model(32, 100);

    model.Input(784, "input layer");
    model.Dense(64, "Hidden layer 1", "relu");
    model.Dense(10, "Hidden layer 2", "none");
    model.Output(10, "Output layer");

    model.compile();
    model.loadData(X, y);
    model.run();
}

