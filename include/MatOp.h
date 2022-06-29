#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

class Node;
class Op {
public:
    Op() = default;

    virtual Node getNewNode();

    virtual vector<MatrixXd> compute(Node &node, vector<MatrixXd> &input_vals);

    virtual vector<Node> gradient(Node &node, Node &output_gradient);

    ~Op() = default;
};