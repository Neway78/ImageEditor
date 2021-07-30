#ifndef OUTILS
#define OUTILS

#include <QImage>

#include <opencv2/opencv.hpp>

QImage Mat2QImage(cv::Mat const& src);

#endif