#include <iostream>
#include <cmath>
#include "Softmax.h"
#include "Node.h"
using namespace std;
using namespace Eigen;

int g_tests_passed = 0, g_tests_failed = 0;

#define TEST(name) cout << "\n[TEST] " << name << "..." << endl
#define CHECK(cond, msg) do { if(!(cond)){ cout << "  FAIL: " << msg << endl; g_tests_failed++; return; } } while(0)
#define PASS() do { cout << "  PASS" << endl; g_tests_passed++; } while(0)

void test_softmax_forward() {
    TEST("Softmax Forward");
    Node A = Variable("logits"), Y = Variable("y_true");
    Node S = softmax_op.getNewNode(A, Y);
    MatrixXd Av(2, 3);
    Av << 1, 2, 3, 4, 5, 6;
    MatrixXd Yv(2, 3);
    Yv << 1, 0, 0, 0, 0, 1;
    map<int, MatrixXd> fd;
    fd[A.hash_code] = Av; fd[Y.hash_code] = Yv;
    Executor exe({S});
    exe.run(fd);
    MatrixXd res = fd[S.hash_code];
    CHECK(abs(res.row(0).sum()-1.0)<1e-6, "Softmax row 0 doesn't sum to 1");
    CHECK(abs(res.row(1).sum()-1.0)<1e-6, "Softmax row 1 doesn't sum to 1");
    CHECK(res(0,2)>res(0,0) && res(0,2)>res(0,1), "Softmax ordering wrong row0");
    CHECK(res(1,2)>res(1,0) && res(1,2)>res(1,1), "Softmax ordering wrong row1");
    PASS();
}

void test_softmax_gradient() {
    TEST("SoftmaxGradient shortcut: d(CE)/d(logits) = softmax - y_true");
    Node A = Variable("logits"), Y = Variable("y_true");
    Node S = softmax_op.getNewNode(A, Y);
    MatrixXd Av(2, 3);
    Av << 1, 2, 3, 4, 5, 6;
    MatrixXd Yv(2, 3);
    Yv << 1, 0, 0, 0, 0, 1;
    map<int, MatrixXd> fd;
    fd[A.hash_code] = Av; fd[Y.hash_code] = Yv;
    Executor exe({S});
    exe.run(fd);

    Node G = softmax_gradient_op.getNewNode(S, Y);
    Executor exe2({G});
    exe2.run(fd);
    MatrixXd grad = fd[G.hash_code];
    MatrixXd expected = fd[S.hash_code] - Yv;
    CHECK((grad-expected).cwiseAbs().maxCoeff()<1e-6, "SoftmaxGradient not y_hat-y");
    cout << "  SoftmaxGradient = softmax - y_true: VERIFIED" << endl;
    PASS();
}

int main() {
    cout << "========================================" << endl;
    cout << "  Softmax Unit Tests" << endl;
    cout << "========================================" << endl;
    test_softmax_forward();
    test_softmax_gradient();
    cout << "\n========================================" << endl;
    cout << "  Results: " << g_tests_passed << " passed, " << g_tests_failed << " failed" << endl;
    cout << "========================================" << endl;
    return g_tests_failed > 0 ? 1 : 0;
}
