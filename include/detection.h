#ifndef DETECTION_H
#define DETECTION_H

#include <QString>

#include <opencv2/opencv.hpp>

#define DEG2RAD 0.017453292519943295

// Filtre de Canny
void convolutionPixelGradient(const cv::Mat &input, cv::Mat &Gx, cv::Mat &Gy, 
                              int x, int y, int nl, int nc, 
                              int masqueX[3][3], int masqueY[3][3]);

void computeGradientPoints(const cv::Mat &G, const cv::Mat &Gx, const cv::Mat &Gy,
                           cv::Mat &gradP1, cv::Mat &gradP2, int nl, int nc);

void detectionMaxima(const cv::Mat &G, const cv::Mat &gradP1, const cv::Mat &gradP2, 
                     cv::Mat &contour, int nl, int nc);

bool checkEdge(const cv::Mat &output, int x, int y, int nl, int nc);


void gradientExtractContour(const cv::Mat &G, cv::Mat &contour, 
                            int seuilBas, int seuilHaut, int nl, int nc);

cv::Mat computeCanny(cv::Mat input, const int maskSize, const double sigma, 
                     const int seuilBas, const int seuilHaut);


// Détecteur de Moravec
float diffIntensiteFenetre(const cv::Mat &input, int nl, int nc, 
                           int u, int v, int halfWindowSize, int x, int y);

cv::Mat computeMoravec(cv::Mat input, const int threshold);


// Détecteur de Harris
cv::Mat computeHarris(cv::Mat input, const int maskSize, 
                      const double sigma, const float R);


// Transformée de Hough pour la détection de lignes
cv::Mat computeHough(cv::Mat input, const int maskSize, const double sigma, 
                     const int seuilBas, const int seuilHaut, const int threshold);


#endif