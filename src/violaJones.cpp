#include "violaJones.h"

cv::Mat haarCascadeClassifier(cv::Mat input)
{
    std::cout << ">> Viola Jones Face Detection:" << "\n\n";

    // load the Haar Cascade Classifier from the xml file
    const char *classifierFile("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml");  
    cv::CascadeClassifier classifier(classifierFile);

    cv::Mat grayscaleImage;
    cv::cvtColor(input, grayscaleImage,  cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> detectedFaces;

    classifier.detectMultiScale(grayscaleImage, detectedFaces, 1.3, 5);

    // Draw each detected faces as a rectangle
    for (auto &&face : detectedFaces) {
        cv::rectangle(input, face, cv::Scalar(0,255,0));
    }
    
    return input;
}