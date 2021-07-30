#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <algorithm>
#include <opencv2/opencv.hpp>

// Segmentation d'Otsu
cv::Mat computeOtsu(cv::Mat input);

float distanceMeans(int mkOld[], int mkNew[], int K);

// Algorithme K-Means
cv::Mat computeKMeans(cv::Mat input, int K, float eps);

#endif

