#ifndef VECTOR_OPS
#define VECTOR_OPS
#include <vector>
using namespace std;
void print ( const vector <float>& m, int n_rows, int n_columns );
int argmax ( const vector <float>& m );
vector<float> random_vector(const int size);
vector <float> operator+(const vector <float>& m1, const vector <float>& m2);
vector <float> operator-(const vector <float>& m1, const vector <float>& m2);
vector <float> operator*(const vector <float>& m1, const vector <float>& m2);
vector <float> operator*(const float m1, const vector <float>& m2);
vector <float> operator/(const vector <float>& m2, const float m1);
vector <float> transform (float *m, const int C, const int R);
vector <float> dot (const vector <float>& m1, const vector <float>& m2, const int m1_rows, const int m1_columns, const int m2_columns);
#endif
