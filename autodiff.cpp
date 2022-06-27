#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include "include/deep_core.h"
#include "include/vector_ops.h"

using namespace std;

class Node(){
    public:
        Node(){

        }
    private:
        vector<Node> input;
        Op op;
};

Class Op(){
    public:
        Op(){

        }

        Node getNewNode(){
            Node newNode = Node();
            newNode.op = this;
            return newNode;
        }

        template<class T>
        virtual vector<T> compute(Node &node, vector<T> &input_vals)=0;

        template<class T>
        virtual vector<T> gradient(Node &node, vector<T> output_gradient)=0;
};

class MatMulOp : public Op(){
    public:
        Node getNewNode(Node &nodeA, Node &nodeB, bool trans_A=false, bool trans_B=false){
            Node new_node = Op::getNewNode();
            new_node

        }
}