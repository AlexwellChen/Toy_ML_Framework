#include <iostream>
#include <cmath>
#include "MatMul.h"
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

void test_matmul_forward() {
    TEST("MatMul Forward");
    Node A = Variable("A"), B = Variable("B");
    Node C = matmul_op.getNewNode(A, B, false, false);
    MatrixXd Av(2, 3), Bv(3, 2);
    Av << 1,2,3, 4,5,6; Bv << 7,8, 9,10, 11,12;
    map<int, MatrixXd> fd;
    fd[A.hash_code] = Av; fd[B.hash_code] = Bv;
    Executor exe({C});
    exe.run(fd);
    MatrixXd res = fd[C.hash_code];
    CHECK(abs(res(0,0)-58)<1e-6 && abs(res(0,1)-64)<1e-6 && abs(res(1,0)-139)<1e-6 && abs(res(1,1)-154)<1e-6, "MatMul result mismatch");
    PASS();
}

void test_matmul_backward() {
    TEST("MatMul Backward (gradient check)");
    Node A = Variable("A"), B = Variable("B");
    Node C = matmul_op.getNewNode(A, B, false, false);
    MatrixXd Av = MatrixXd::Random(3, 4), Bv = MatrixXd::Random(4, 2);
    map<int, MatrixXd> fd;
    fd[A.hash_code] = Av; fd[B.hash_code] = Bv;
    Executor exe({C});
    CHECK(check_grad(C, A, fd, exe), "Gradient check failed for A");
    CHECK(check_grad(C, B, fd, exe), "Gradient check failed for B");
    PASS();
}

int main() {
    cout << "========================================" << endl;
    cout << "  MatMul Unit Tests" << endl;
    cout << "========================================" << endl;
    test_matmul_forward();
    test_matmul_backward();
    cout << "\n========================================" << endl;
    cout << "  Results: " << g_tests_passed << " passed, " << g_tests_failed << " failed" << endl;
    cout << "========================================" << endl;
    return g_tests_failed > 0 ? 1 : 0;
}
