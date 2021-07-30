#include "outils.h"

// Ensemble de fonctions utilitaires

QImage Mat2QImage(cv::Mat const& src)
{
    cv::Mat temp;
    if (src.channels() == 1) {
        cv::cvtColor(src, temp, cv::COLOR_GRAY2RGB);
    } else {
        cv::cvtColor(src, temp, cv::COLOR_BGR2RGB);
    }
    QImage dest((const uchar *) temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
    dest.bits();
    return dest;
}
