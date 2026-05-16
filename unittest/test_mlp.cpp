#include <iostream>
#include "Model.h"
using namespace std;

int g_tests_passed = 0, g_tests_failed = 0;

#define TEST(name) cout << "\n[TEST] " << name << "..." << endl
#define CHECK(cond, msg) do { if(!(cond)){ cout << "  FAIL: " << msg << endl; g_tests_failed++; return; } } while(0)
#define PASS() do { cout << "  PASS" << endl; g_tests_passed++; } while(0)

void test_mlp_training() {
    TEST("MLP Training - Loss Decreases");

    int batch_size = 32;
    Model model(batch_size, 50, 0.05f, 1.0f, 9999);
    model.csv_filename = "output/test_mlp_metrics.csv";
    model.log_interval = 50;

    model.Input(784, "input");
    model.Dense(32, "hidden1", "relu");
    model.Dense(16, "hidden2", "relu");
    model.Output(10, "output");

    model.compile();
    model.loadData("train.txt");
    model.run();

    float final_acc = model.evaluate(model.X_train, model.y_train);
    cout << "  Final train accuracy: " << final_acc << endl;
    CHECK(final_acc > 0.15, "MLP accuracy did not improve beyond random baseline");
    PASS();
}

int main() {
    cout << "========================================" << endl;
    cout << "  MLP Integration Test" << endl;
    cout << "========================================" << endl;
    test_mlp_training();
    cout << "\n========================================" << endl;
    cout << "  Results: " << g_tests_passed << " passed, " << g_tests_failed << " failed" << endl;
    cout << "========================================" << endl;
    return g_tests_failed > 0 ? 1 : 0;
}
