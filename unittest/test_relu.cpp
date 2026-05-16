#include <iostream>
#include <cmath>
#include "Relu.h"
#include "Node.h"
using namespace std;
using namespace Eigen;

const double EPSILON = 1e-5;
const double TOLERANCE = 1e-4;
int g_tests_passed = 0, g_tests_failed = 0;

#define TEST(name) cout << "\n[TEST] " << name << "..." << endl
#define CHECK(cond, msg) do { if(!(cond)){ cout << "  FAIL: " << msg << endl; g_tests_failed++; return; } } while(0)
#define PASS() do { cout << "  PASS" << endl; g_tests_passed++; } while(0)

bool grad_ok(const MatrixXd& a, const MatrixXd& n, double tol = TOLERANCE) {
    double d = (a - n).cwiseAbs().maxCoeff();
    double r = 0.0, denom = (a.cwiseAbs() + n.cwiseAbs()).maxCoeff();
    if (denom > 1e-10) r = (a - n).cwiseAbs().maxCoeff() / denom;
    cout << "    max_abs_diff=" << d << " rel_error=" << r << endl;
    return d < tol || r < tol;
}

double eval_scalar(Node& out, map<int, MatrixXd>& fd, Executor& exe) {
    exe.run(fd);
    return fd[out.hash_code].sum();
}

bool check_grad(Node& out, Node& w, map<int, MatrixXd>& fd, Executor& exe, double tol = TOLERANCE) {
    vector<Node> inputs = {w};
    vector<Node> grads = gradients(out, inputs);
    vector<Node> eval_list = {out};
    eval_list.insert(eval_list.end(), grads.begin(), grads.end());
    Executor exe2(eval_list);
    exe2.run(fd);
    MatrixXd analytical = fd[grads[0].hash_code];

    MatrixXd numerical(analytical.rows(), analytical.cols());
    MatrixXd& W = fd[w.hash_code];
    for (int r = 0; r < W.rows(); r++)
        for (int c = 0; c < W.cols(); c++) {
            double orig = W(r, c);
            W(r, c) = orig + EPSILON;
            double fp = eval_scalar(out, fd, exe);
            W(r, c) = orig - EPSILON;
            double fm = eval_scalar(out, fd, exe);
            W(r, c) = orig;
            numerical(r, c) = (fp - fm) / (2 * EPSILON);
        }
    return grad_ok(analytical, numerical, tol);
}

void test_relu_forward() {
    TEST("ReLU Forward");
    Node A = Variable("A");
    Node R = relu_op.getNewNode(A);
    MatrixXd Av(2, 3);
    Av << -1, 2, -3, 4, -5, 6;
    map<int, MatrixXd> fd;
    fd[A.hash_code] = Av;
    Executor exe({R});
    exe.run(fd);
    MatrixXd res = fd[R.hash_code];
    CHECK(res(0,0)==0 && res(0,1)==2 && res(1,2)==6, "ReLU result mismatch");
    PASS();
}

void test_relu_backward() {
    TEST("ReLU Backward (gradient check)");
    Node A = Variable("A");
    Node R = relu_op.getNewNode(A);
    MatrixXd Av = MatrixXd::Random(4, 3);
    map<int, MatrixXd> fd;
    fd[A.hash_code] = Av;
    Executor exe({R});
    CHECK(check_grad(R, A, fd, exe, 1e-3), "Gradient check failed");
    PASS();
}

int main() {
    cout << "========================================" << endl;
    cout << "  ReLU Unit Tests" << endl;
    cout << "========================================" << endl;
    test_relu_forward();
    test_relu_backward();
    cout << "\n========================================" << endl;
    cout << "  Results: " << g_tests_passed << " passed, " << g_tests_failed << " failed" << endl;
    cout << "========================================" << endl;
    return g_tests_failed > 0 ? 1 : 0;
}
