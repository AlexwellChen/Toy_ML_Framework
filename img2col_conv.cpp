#include <iostream>
#include <Eigen/Dense>
#include <utility>
#include <Eigen/Core>
#include <vector>
#include <utility>

using namespace std;
using namespace Eigen;

/*
 * input按照NCHW分布，最外层vector储存每一张图片，Matrix存储图片
 * kernel也按照NCHW分布，最外层vector存储卷积核，Matrix存储对应的卷积核
 * 为了简单起见，仅实现单通道卷积，C=1
 */
vector<MatrixXd> img2col2D(const MatrixXd &input, const vector<MatrixXd> &kernels, const int &kernel_H, const int &kernel_W, const int &paddle, const int &stride){
    vector<MatrixXd> feature_maps;
    long img_H = input.rows();
    long img_W = input.cols();
    /*
     * 对input img进行paddle
     */
    // 生成一个全0的paddle矩阵
    MatrixXd paddle_img = MatrixXd::Zero(img_H + 2 * paddle, img_W + 2 * paddle);
    // 将图像放置在paddle矩阵的中心
    paddle_img.block(paddle, paddle, img_H, img_W) = input;

    img_H = paddle_img.rows();
    img_W = paddle_img.cols();

    // 生成输入矩阵
    int input_matrix_W = kernel_H * kernel_W;
    int input_matrix_H = int(((img_H - kernel_H + 1) / stride) * ((img_W - kernel_W + 1) / stride));
    MatrixXd input_matrix(input_matrix_H, input_matrix_W);
    int row_idx = 0;
    int col_idx = 0;
    for(int i = 0; i < img_H - kernel_H + 1; i += stride){
        for(int j = 0; j < img_W - kernel_W + 1; j += stride){
            col_idx = 0;
            for(int m = i; m < i + kernel_H; ++m){
                for(int n = j; n < j + kernel_W; ++n){
                    input_matrix(row_idx, col_idx) = paddle_img(m, n);
                    col_idx++;
                }
            }
            row_idx++;
        }
    }

    //生成卷积矩阵
    int kernel_nums = kernels.size();
    MatrixXd kernel_matrix(kernel_W * kernel_H, kernel_nums);
    for(int i = 0; i < kernel_nums; ++i){
        for(int m = 0; m < kernel_H; ++m){
            for(int n = 0; n < kernel_W; ++n){
                kernel_matrix(m * kernel_H + n, i) = kernels[i](m, n);
            }
        }
    }

    // 复原feature map
    MatrixXd feature_matrix = input_matrix * kernel_matrix;
    int feature_H = int(paddle_img.rows()) - kernel_H + 1;
    int feature_W = int(paddle_img.cols()) - kernel_W + 1;
    for(int i = 0; i < kernel_nums; ++i){
        MatrixXd feature = feature_matrix.col(i);
        feature.resize(feature_H, feature_W);
        feature_maps.emplace_back(feature);
    }
    return std::move(feature_maps);
}

MatrixXd MaxPooling2D(const MatrixXd &input, const int filter_H, const int filter_W, const int stride){
    const int img_H = int(input.rows());
    const int img_W = int(input.cols());
    const int pooling_H = (img_H - filter_H)/stride + 1;
    const int pooling_W = (img_W - filter_W)/stride + 1;
    MatrixXd pooling_img(pooling_H, pooling_W);
    vector<pair<int, int>> pooling_loc(pooling_H * pooling_W);
    for(int i = 0; i < img_H; i += stride){
        for(int j = 0; j < img_W; j += stride){
            int pooling_row;
            int pooling_col;
            float max_val = input.block(i, j, filter_H, filter_W).maxCoeff(&pooling_row, &pooling_col);
            pooling_img(i/2, j/2) = max_val;
            pooling_loc[i/2 * pooling_H + j/2] = pair<int, int>(pooling_row, pooling_col);
        }
    }
    return std::move(pooling_img);
}

int main(){
    MatrixXd input_img(10, 10);
    input_img = MatrixXd::Random(10, 10);
    cout << "Original image:" << endl;
    cout << input_img << endl;
//    vector<MatrixXd> kernels;
//    int kernel_num = 8;
//    for(int i = 0; i < kernel_num; ++i){
//        MatrixXd kernel;
//        kernel = MatrixXd::Random(3, 3);
//        kernels.emplace_back(kernel);
//    }
//    cout << "Original image:" << endl;
//    cout << input_img << endl;
//    vector<MatrixXd> feature_maps = img2col2D(input_img, kernels, 3, 3, 0, 1);
//    cout << "Feature map:" << endl;
//    for(MatrixXd &feature : feature_maps){
//        cout << "----------------------" << endl;
//        cout << feature << endl;
//    }
    cout << "Pooling image:" << endl;
    MatrixXd pooling_img = MaxPooling2D(input_img, 2, 2, 2);
    cout << pooling_img << endl;
}