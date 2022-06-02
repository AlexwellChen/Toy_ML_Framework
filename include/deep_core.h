#ifndef DEEP_CORE
#define DEEP_CORE
#include <vector>
using namespace std;
vector <float> softmax (const vector <float>& z, const int dim);    
vector <float> sigmoid_d (const vector <float>& m1);
vector <float> sigmoid (const vector <float>& m1);
vector <float> relu(const vector <float>& z);
vector <float> reluPrime (const vector <float>& z);
#endif
