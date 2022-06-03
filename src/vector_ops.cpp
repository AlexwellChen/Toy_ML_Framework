#include <random>
#include <chrono>
#include <algorithm>
#include <vector>
#include <iostream>
#include <cblas.h>
#include <arm_neon.h>
#include "vector_ops.h" 

using namespace std;

void print ( const vector <float>& m, int n_rows, int n_columns ) {
    
    /*  "Couts" the input vector as n_rows x n_columns matrix.
     Inputs:
     m: vector, matrix of size n_rows x n_columns
     n_rows: int, number of rows in the left matrix m1
     n_columns: int, number of columns in the left matrix m1
     */
    
    for( int i = 0; i != n_rows; ++i ) {
        for( int j = 0; j != n_columns; ++j ) {
            cout << m[ i * n_columns + j ] << " ";
        }
        cout << '\n';
    }
    cout << endl;
}

int argmax ( const vector <float>& m ) {
    return distance(m.begin(), max_element(m.begin(), m.end()));
}


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

vector <float> operator+(const vector <float>& m1, const vector <float>& m2){
    
    /*  Returns the elementwise sum of two vectors.
     Inputs:
     m1: a vector
     m2: a vector
     Output: a vector, sum of the vectors m1 and m2.
     */
    
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> sum (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        sum[i] = m1[i] + m2[i];
    };
    
    return sum;
}

vector <float> operator-(const vector <float>& m1, const vector <float>& m2){
    
    /*  Returns the difference between two vectors.
     Inputs:
     m1: vector
     m2: vector
     Output: vector, m1 - m2, difference between two vectors m1 and m2.
     */
    
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> difference (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        difference[i] = m1[i] - m2[i];
    };
    
    return difference;
}

vector <float> operator*(const vector <float>& m1, const vector <float>& m2){
    
    /*  Returns the product of two vectors (elementwise multiplication).
     Inputs:
     m1: vector
     m2: vector
     Output: vector, m1 * m2, product of two vectors m1 and m2
     */
    
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> product (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        product[i] = m1[i] * m2[i];
    };
    
    return product;
}

vector <float> operator*(const float m1, const vector <float>& m2){
    
    /*  Returns the product of a float and a vectors (elementwise multiplication).
     Inputs:
     m1: float
     m2: vector
     Output: vector, m1 * m2, product of two vectors m1 and m2
     */
    
    const unsigned long VECTOR_SIZE = m2.size();
    vector <float> product (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        product[i] = m1 * m2[i];
    };
    
    return product;
}

vector <float> operator/(const vector <float>& m2, const float m1){
    
    /*  Returns the product of a float and a vectors (elementwise multiplication).
     Inputs:
     m1: float
     m2: vector
     Output: vector, m1 * m2, product of two vectors m1 and m2
     */
    
    const unsigned long VECTOR_SIZE = m2.size();
    vector <float> product (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        product[i] = m2[i] / m1;
    };
    
    return product;
}

vector <float> transpose (const vector <float>& m, const int C, const int R) {
    
    /*  Returns a transpose matrix of input matrix.
     Inputs:
     m: vector, input matrix
     C: int, number of columns in the input matrix
     R: int, number of rows in the input matrix
     Output: vector, transpose matrix mT of input matrix m
     */
    
    vector <float> mT (C*R);
    
    for(unsigned n = 0; n != C*R; n++) {
        unsigned i = n/C;
        unsigned j = n%C;
        mT[n] = m[R*j + i];
    }
    
    return mT;
}

void matrix_multiply_neon(const float32_t  *A, const float32_t  *B, float32_t *C, const uint32_t n, const uint32_t m, const uint32_t k) {
    /* 
     * Multiply matrices A and B, store the result in C. 
     * It is the user's responsibility to make sure the matrices are compatible.
     */     

    int A_idx;
    int B_idx;
    int C_idx;
        
    // these are the columns of a 4x4 sub matrix of A
    float32x4_t A0;
    float32x4_t A1;
    float32x4_t A2;
    float32x4_t A3;
        
    // these are the columns of a 4x4 sub matrix of B
    float32x4_t B0;
    float32x4_t B1;
    float32x4_t B2;
    float32x4_t B3;
        
    // these are the columns of a 4x4 sub matrix of C
    float32x4_t C0;
    float32x4_t C1;
    float32x4_t C2;
    float32x4_t C3;
        
    for (int i_idx=0; i_idx<n; i_idx+=4) {
        for (int j_idx=0; j_idx<m; j_idx+=4){
            // zero accumulators before matrix op
            C0=vmovq_n_f32(0);
            C1=vmovq_n_f32(0);
            C2=vmovq_n_f32(0); 
            C3=vmovq_n_f32(0);
            for (int k_idx=0; k_idx<k; k_idx+=4){
                // compute base index to 4x4 block
                A_idx = i_idx + n*k_idx;
                B_idx = k*j_idx + k_idx;

                // load most current a values in row
                A0=vld1q_f32(A+A_idx);
                A1=vld1q_f32(A+A_idx+n);
                A2=vld1q_f32(A+A_idx+2*n);
                A3=vld1q_f32(A+A_idx+3*n);

                // multiply accumulate 4x1 blocks, i.e. each column C
                B0=vld1q_f32(B+B_idx);
                C0=vfmaq_laneq_f32(C0,A0,B0,0);
                C0=vfmaq_laneq_f32(C0,A1,B0,1);
                C0=vfmaq_laneq_f32(C0,A2,B0,2);
                C0=vfmaq_laneq_f32(C0,A3,B0,3);

                B1=vld1q_f32(B+B_idx+k);
                C1=vfmaq_laneq_f32(C1,A0,B1,0);
                C1=vfmaq_laneq_f32(C1,A1,B1,1);
                C1=vfmaq_laneq_f32(C1,A2,B1,2);
                C1=vfmaq_laneq_f32(C1,A3,B1,3);

                B2=vld1q_f32(B+B_idx+2*k);
                C2=vfmaq_laneq_f32(C2,A0,B2,0);
                C2=vfmaq_laneq_f32(C2,A1,B2,1);
                C2=vfmaq_laneq_f32(C2,A2,B2,2);
                C2=vfmaq_laneq_f32(C2,A3,B3,3);

                B3=vld1q_f32(B+B_idx+3*k);
                C3=vfmaq_laneq_f32(C3,A0,B3,0);
                C3=vfmaq_laneq_f32(C3,A1,B3,1);
                C3=vfmaq_laneq_f32(C3,A2,B3,2);
                C3=vfmaq_laneq_f32(C3,A3,B3,3);
            }
            //Compute base index for stores
            C_idx = n*j_idx + i_idx;
            vst1q_f32(C+C_idx, C0);
            vst1q_f32(C+C_idx+n, C1);
            vst1q_f32(C+C_idx+2*n,C2);
            vst1q_f32(C+C_idx+3*n,C3);
        }
    }
}

vector <float> dot (const vector <float>& m1, const vector <float>& m2, const int m1_rows, const int m1_columns, const int m2_columns, const int type=0) {
    
    /*  Returns the product of two matrices: m1 x m2.
     Inputs:
     m1: vector, left matrix of size m1_rows x m1_columns
     m2: vector, right matrix of size m1_columns x m2_columns (the number of rows in the right matrix
     must be equal to the number of the columns in the left one)
     m1_rows: int, number of rows in the left matrix m1
     m1_columns: int, number of columns in the left matrix m1
     m2_columns: int, number of columns in the right matrix m2
     Output: vector, m1 * m2, product of two vectors m1 and m2, a matrix of size m1_rows x m2_columns
     */
    
    vector <float> output (m1_rows*m2_columns); // 结果矩阵
    float r;
    int N = m1_rows;
    int M = m2_columns;
    int K = m1_columns;
    switch (type) {
        case 0:
            for( int row = 0; row < m1_rows; ++row ) {
                for( int col = 0; col < m2_columns; ++col ) {
                    for( int k = 0; k < m1_columns; ++k ) {
                        output[ row * m2_columns + col ] += m1[ row * m1_columns + k ] * m2[ k * m2_columns + col ];
                    }
                }
            }
            break;
        case 1:
            // loop interchange
            for( int row = 0; row < m1_rows; ++row ) {
                for( int k = 0; k < m1_columns; ++k ) {
                    r = m1[row * m1_columns + k];
                    for( int col = 0; col < m2_columns; ++col ) {
                        output[ row * m2_columns + col ] += r * m2[k * m2_columns + col];
                    }
                }
            }
            break;
        case 2:
        {
            // loop interchange + loop tiling
            const int block_size = 128; // 64bytes / 4bytes(a float)
            for( int i = 0; i < N; i += block_size){
                int imax =(i + block_size) > M ? M : (i + block_size);
                for( int j = 0; j < M; j += block_size){
                    int jmax = (j + block_size) > N ? N : (j + block_size);
                    for(int k = 0; k < K; k += block_size){
                        int kmax = (k + block_size) > K ? K : (k + block_size);
                        for(int ii = i; ii < imax; ii++){
                            for(int ik = k; ik < kmax; ik++){
                                r = m1[ii * K + ik];
                                for(int ij = j; ij < jmax; ij++){
                                    output[ii * N + ij] += r * m2[ik * N + ij];
                                }
                            }
                        }
                    }
                }
            }
            break;
        }
        case 3:
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, M, K, 1.0, m1.data(), m1_columns, m2.data(), m2_columns, 0.0, output.data(), m2_columns);
            break;
        case 4:
            // loop interchange + loop tiling + neon
            if(M%4==0 && N%4==0 && K%4==0){
                matrix_multiply_neon(m1.data(), m2.data(), output.data(), N, M, K);
            }else{
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, M, K, 1.0, m1.data(), m1_columns, m2.data(), m2_columns, 0.0, output.data(), m2_columns);
            }
            break;
    }
    return output;
}

