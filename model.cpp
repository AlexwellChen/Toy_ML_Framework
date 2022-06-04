#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
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

class Model{
    private:
        vector<Layer*> layers;
        vector<vector<float>> mats;

        vector<float> random_vector(const int size)
        {
            /*  Generates a random vector with uniform distribution
            Inputs:
            size: the vector size
            */
            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<> distribution(0.0, 0.05);
            static default_random_engine generator;
            
            vector<float> data(size);
            generate(data.begin(), data.end(), [&]() { return distribution(generator); });
            return data;
        }


    public:
        Model(){
            // Nothing
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
                if(i == 2){
                    break;
                }
                cout << "-----------------------------" << endl;
                cout << "Layer " << i << " and Layer " << j << ":" << endl;
                int n = layers[i]->getNodeNum();
                int m = layers[j]->getNodeNum();
                print_mat(mats[i], n, m);
            }
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