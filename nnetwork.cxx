//
//  nn_mpi.cpp
//
//  To compile: mpicxx -std=c++11 -O3 -fopenmp -o train_mpi
//  To run: ./train_mpi

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include "deep_core.h"
#include "vector_ops.h"

#define INIT 0 // 10000:
#define INTERCHANGE 1 // 10000:
#define TILING_INTERCHANGE 2 // 10000:
#define BLAS 3 // 10000:
#define NEON 4
#define MULTI_THREAD 5


#define ITER 10000

#define DOT BLAS
#define BATCH_SIZE 32

using namespace std;

vector<string> split(const string &s, char delim) {
    stringstream ss(s);
    string item;
    vector<string> tokens;
    while (getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}

int main(int argc, char * argv[]) {
    string line;
    vector<string> line_v;
    int len, mpirank = 0;
    cout << "Loading data ...\n";
    vector<float> X_train;
    vector<float> y_train;
    ifstream myfile ("train.txt");
    if (myfile.is_open())
    {
        while ( getline (myfile,line) )
        {
            line_v = split(line, '\t');
            int digit = strtof((line_v[0]).c_str(),0);
            for (unsigned i = 0; i < 10; ++i) {
                if (i == digit)
                {
                    y_train.push_back(1.);
                }
                else y_train.push_back(0.);
            }
            
            int size = static_cast<int>(line_v.size());
            for (unsigned i = 1; i < size; ++i) {
                X_train.push_back(strtof((line_v[i]).c_str(),0));
            }
        }
        X_train = X_train/255.0;
        myfile.close();
    }
    
    else cout << "Unable to open file" << '\n';
    
    int xsize = static_cast<int>(X_train.size());
    int ysize = static_cast<int>(y_train.size());
    
    // Some hyperparameters for the NN
    float lr = .01/BATCH_SIZE;
    // Random initialization of the weights
    // Input layer 784 vector
    vector <float> W1 = random_vector(784*128);
    // First hidden layer 128
    vector <float> W2 = random_vector(128*64);
    // Second hidden layer 64
    vector <float> W3 = random_vector(64*10);
    // Output layer 10
    
    
    std::chrono::time_point<std::chrono::system_clock> t1,t2,t_start,t_end;
    cout << "Training the model ...\n";
    t_start = std::chrono::system_clock::now();
    for (unsigned i = 0; i < ITER; ++i) {
        t1 = std::chrono::system_clock::now();
        // Building batches of input variables (X) and labels (y)
        int randindx = rand() % (42000-BATCH_SIZE);
        vector<float> b_X;
        vector<float> b_y;
        for (unsigned j = randindx*784; j < (randindx+BATCH_SIZE)*784; ++j){
            b_X.push_back(X_train[j]);
        }
        for (unsigned k = randindx*10; k < (randindx+BATCH_SIZE)*10; ++k){
            b_y.push_back(y_train[k]);
        }
        
        // Feed forward
        vector<float> a1 = relu(dot( b_X, W1, BATCH_SIZE, 784, 128, DOT));
        vector<float> a2 = relu(dot( a1, W2, BATCH_SIZE, 128, 64, DOT));
        vector<float> yhat = softmax(dot( a2, W3, BATCH_SIZE, 64, 10, DOT), 10);
        
        // Back propagation
        vector<float> dyhat = (yhat - b_y); // softmax derivative
        // dW3 = a2.T * dyhat
        vector<float> dW3 = dot(transpose(a2, BATCH_SIZE, 64 ), dyhat, 64, BATCH_SIZE, 10, DOT);
        // dz2 = dyhat * W3.T * relu'(a2)
        vector<float> dz2 = dot(dyhat, transpose(W3, 64, 10 ), BATCH_SIZE, 10, 64, DOT) * reluPrime(a2);
        // dW2 = a1.T * dz2
        vector<float> dW2 = dot(transpose(a1, BATCH_SIZE, 128 ), dz2, 128, BATCH_SIZE, 64, DOT);
        // dz1 = dz2 * W2.T * relu'(a1)
        vector<float> dz1 = dot(dz2, transpose(W2, 128, 64 ), BATCH_SIZE, 64, 128, DOT) * reluPrime(a1);
        // dW1 = X.T * dz1
        vector<float> dW1 = dot(transpose(b_X, BATCH_SIZE, 784 ), dz1, 784, BATCH_SIZE, 128, DOT);
        
        
        // Updating the parameters
        W3 = W3 - lr * dW3;
        W2 = W2 - lr * dW2;
        W1 = W1 - lr * dW1;
        

        if ((mpirank == 0 && (i+1) % 1000 == 0)){
            cout << "Predictions:" << "\n";
            print ( yhat, 10, 10 );
            cout << "Ground truth:" << "\n";
            print ( b_y, 10, 10 );
            vector<float> loss_m = yhat - b_y; // b_y is true label
            float loss = 0.0;
            for (unsigned k = 0; k < BATCH_SIZE*10; ++k){
                loss += loss_m[k]*loss_m[k];
            }
            t2 = std::chrono::system_clock::now();
            chrono::duration<double> elapsed_seconds = t2-t1;
            double ticks = elapsed_seconds.count();
            cout << "Iteration #: "  << i << endl;
            cout << "Iteration Time: "  << ticks << "s" << endl;
            cout << "Loss: " << loss/(BATCH_SIZE*10) << endl;
            cout << "*******************************************" << endl;
        };
    };
    t_end = std::chrono::system_clock::now();
    chrono::duration<double> duration = t_end-t_start;
    double ticks = duration.count();
    cout << "Duration Time: "  << ticks << "s" << endl;
    return 0;
}
