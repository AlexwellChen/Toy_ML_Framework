#include <iostream>
#include "Constant.h"
#include "Node.h"
using namespace std;
using namespace Eigen;

int g_tests_passed = 0, g_tests_failed = 0;
#define TEST(name) cout << "\n[TEST] " << name << "..." << endl
#define CHECK(cond, msg) do { if(!(cond)){ cout << "  FAIL: " << msg << endl; g_tests_failed++; return; } } while(0)
#define PASS() do { cout << "  PASS" << endl; g_tests_passed++; } while(0)

void test_variable() {
    TEST("Variable / Placeholder");
    Node v = Variable("test_var");
    CHECK(v.name == "test_var", "Variable name mismatch");
    CHECK(v.isPlaceHolder == true, "Variable not a placeholder");
    CHECK(v.hash_code != -1, "Variable hash_code not set");
    PASS();
}

void test_oneslike() {
    TEST("OnesLike Forward");
    Node A = Variable("A");
    Node O = ones_like_op.getNewNode(A);
    MatrixXd Av(3, 2); Av << 1,2,3,4,5,6;
    map<int, MatrixXd> fd; fd[A.hash_code] = Av;
    Executor exe({O}); exe.run(fd);
    MatrixXd res = fd[O.hash_code];
    CHECK(res.rows()==3 && res.cols()==2, "Shape mismatch");
    CHECK(res(0,0)==1 && res(2,1)==1, "Not all ones");
    PASS();
}

void test_zeroslike() {
    TEST("ZerosLike Forward");
    Node A = Variable("A");
    Node Z = zeros_like_op.getNewNode(A);
    MatrixXd Av(3, 2); Av << 1,2,3,4,5,6;
    map<int, MatrixXd> fd; fd[A.hash_code] = Av;
    Executor exe({Z}); exe.run(fd);
    MatrixXd res = fd[Z.hash_code];
    CHECK(res.rows()==3 && res.cols()==2, "Shape mismatch");
    CHECK(res(0,0)==0 && res(2,1)==0, "Not all zeros");
    PASS();
}

int main() {
    cout << "========================================" << endl;
    cout << "  Constant / Placeholder Unit Tests" << endl;
    cout << "========================================" << endl;
    test_variable();
    test_oneslike();
    test_zeroslike();
    cout << "\n========================================" << endl;
    cout << "  Results: " << g_tests_passed << " passed, " << g_tests_failed << " failed" << endl;
    cout << "========================================" << endl;
    return g_tests_failed > 0 ? 1 : 0;
}
