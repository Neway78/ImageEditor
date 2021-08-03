#ifndef FILTER_H
#define FILTER_H

#include <cmath>
#include <opencv2/opencv.hpp>

#include "openclProgram.h"
#include "kernel.h"

#define PI 3.141592653589793
#define PI_SQUARE 9.869604401089358

// Convolution spatiale
uchar convolutionPixel(const cv::Mat &input, const int maskSize, const double sigma, const int x, const int y);

cv::Mat gaussConvolutionFilter(cv::Mat input, const int maskSize, const double sigma);

// Convolution spatiale Gradient
float convolutionPixelGrad(const cv::Mat &input, const int maskSize, const double sigma, const int x, const int y);

cv::Mat gaussConvolutionFilterGrad(cv::Mat input, const int maskSize, const double sigma);

// Convolution par FFT
void fftshift(cv::Mat fourierMat);

cv::Mat fftConvolution(cv::Mat input, const double sigma);

// Filtrage laplacien
cv::Mat laplacianGPUFilter(unsigned char *input, int width, int height, OpenCLProgram &ourProgram);

// Filtre median
cv::Mat computeMedian(cv::Mat input, const int size);

// Filtre bilateral
cv::Mat computeBilateral(cv::Mat input, const double sigma1, const double sigma2, const int nbIter);

#endif
