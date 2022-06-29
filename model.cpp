#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include "include/deep_core.h"
#include "include/vector_ops.h"

using namespace std;


class Layer{
    public:

        Layer(int &nodeNum){
            if(nodeNum%4){
                int d = nodeNum%4;
                node_number = nodeNum + d; // padding, 使得矩阵边长可以被4整除
                valid_node_number = nodeNum;
            }else{
                node_number = nodeNum;
                valid_node_number = nodeNum;
            }
        }

        int getNodeNum(){
            return node_number;
        }

        int getValidNodeNum(){
            return valid_node_number;
        }

    private:
        int node_number;
        int valid_node_number;
};

class InputLayer : public Layer{
    public:
        InputLayer(int &node_number, int &batchSize){
            Layer(inputSize);
            batch = batchSize;
        }
        void loadData(vector<float> &X_train, int &randindx){
            int inputSize = getNodeNum();
            for (unsigned j = randindx*inputSize; j < (randindx+batchSize)*inputSize; ++j){
                inputMat.push_back(X_train[j]);
            }

        }
    private:
        vector<float> inputMat;
        int batch;
};

class Dense : public Layer{
    public:
        Dense(int inputSize, int batchSize){
            Layer(inputSize);
            vector<float> result = random_vector(batchSize*inputSize);
        }
    private:
        vector<float> W;
        vector<float> result;
}

class Model{
    private:
        vector<Layer*> layers;
        vector<vector<float>> mats;
        int batchSize;
        float lr;
        vector<float> X_train;
        vector<float> y_train;

    public:
        Model(int _batchSize=32){
            batchSize = _batchSize;
            lr = 0.01/_batchSize;
            loadDataSet();
        }

        void add(int nodeNum){
            Layer *new_layer = new Layer(nodeNum);
            layers.push_back(new_layer);
        }

        int compile(){
            if(layers.size() < 2){
                cout << "Too few layers!" << endl;
                return 0;
            }
            int layer_num = layers.size();
            int i = 0;
            int j = 1;
            while(j < layer_num){
                int n = layers[i]->getNodeNum();
                int m = layers[j]->getNodeNum();
                vector<float> W = random_vector(n*m);
                mats.push_back(W);
                i++;
                j++;
            }
            return 1;
        }

        void print_mat ( const vector <float>& m, int n_rows, int n_columns ) {
            /*  "Couts" the input vector as n_rows x n_columns matrix.
            Inputs:
            m: vector, matrix of size n_rows x n_columns
            n_rows: int, number of rows in the left matrix m1
            n_columns: int, number of columns in the left matrix m1
            */
            for( int i = 0; i < n_rows; ++i ) {
                for( int j = 0; j < n_columns; ++j ) {
                    cout << m[ i * n_columns + j ] << " ";
                }
                cout << '\n';
            }
            cout << endl;
        }

        void showNetwork(){
            int mat_num = mats.size();
            cout << "Matrix number: "<< mat_num << endl;
            for(int i = 0, j = 1; i < mat_num; i++, j++){
                cout << "-----------------------------" << endl;
                cout << "Layer " << i << " and Layer " << j << ":" << endl;
                int n = layers[i]->getNodeNum();
                int m = layers[j]->getNodeNum();
                print_mat(mats[i], n, m);
            }
        }

        void loadDataSet(){
            string line;
            vector<string> line_v;
            int len, mpirank = 0;
            cout << "Loading data ...\n";
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
        }

};

int main(int argc, char * argv[]){
    Model model = Model();
    model.add(16);
    model.add(8);
    model.add(4);
    model.compile();
    model.showNetwork();
}