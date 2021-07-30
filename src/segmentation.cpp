#include <iostream>
#include <cmath>

#include "segmentation.h"

cv::Mat computeOtsu(cv::Mat input)
{
    std::cout << ">> Otsu's Segmentation:" << "\n";

    // Conversion de l'image en niveaux de gris
    cv::cvtColor(input, input, cv::COLOR_BGR2GRAY);

    // Egalisation de l'histogramme de l'image
    cv::equalizeHist(input, input);

    // nombre de niveau de gris
    const int numberGray(256);

    // nombre de pixels de l'image
    int nb_pixels = input.rows * input.cols;

    // initialisation de l'histogramme de l'image
    int histImage[numberGray];
    for (int i = 0; i < numberGray; i++) {
        histImage[i] = 0;
    }

    // valeur du pixel de l'image
    int valPixel;

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            valPixel = input.at<uchar>(i,j);
            histImage[valPixel]++;
        }
    }

    // Initialisation du tableau des probailités de chaque niveau de gris
    double probaImage[numberGray];

    for (int i = 0; i < 256; i++) {
        probaImage[i] = histImage[i] / (double)nb_pixels;
    }

    /* Notations :
    wi       : probabilité d'être dans la classe i
    meancr   : somme croissante des 'ip(i)' pour les calculs
    meandr   : somme décroisante des 'ipi)' pour les calculs
    mui      : moyenne de la classe i
    sigmab_t : variance inter-classe compte tenu du seuil t 
    */

    double w0, w1;
    double meancr, meandcr;
    double mu0, mu1;
    double sigmab_t[numberGray];

    w0 = 0.0; w1 = 1.0;
    meancr = 0.0; meandcr = 0.0; 
    for (int i = 0; i < numberGray; i++) {
        meandcr += i * probaImage[i];
    }
    mu0 = 0.0; mu1 = meandcr;

    for (int t = 1; t < numberGray+1; t++) {
        w0 += probaImage[t-1];
        w1 -= probaImage[t-1];
        meancr += (t-1) * probaImage[t-1];
        meandcr -= (t-1) * probaImage[t-1];
        mu0 = meancr / w0;
        mu1 = meandcr / w1;
        sigmab_t[t-1] = w0*w1*pow((mu0-mu1),2);
    }

    // Le seuil est le maximum de la variance inter-classe

    int threshold = 0;
    for (int t = 1; t < numberGray; t++) {
        if (sigmab_t[t] > sigmab_t[threshold]) {
            threshold = t;
        }
    }

    std::cout << "\tThreshold : t = " << threshold << "\n\n";

    // Ajustement de tous les pixels de l'image
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            input.at<uchar>(i,j) = (input.at<uchar>(i,j) <= threshold) ? 0 : 255;
        }
    }

    return input;
};

// Ecart des moyennes aux moyennes précédentes
float distanceMeans(float mkOld[], float mkNew[], int K)
{
    float ecart = 0.0;
    for (int k = 0; k < K; k++) {
        ecart += (mkNew[k]- mkOld[k]) * (mkNew[k] - mkOld[k]);
    }
    ecart = sqrt(ecart) / K;
    return ecart;
}

// Algorithme K-means
cv::Mat computeKMeans(cv::Mat input, int K, float eps) 
{
    printf(">> K-means Algorithm: K = %i, epsilon = %f\n\n", K, eps);
 
    // Dimensions de l'image
    int nl = input.rows;
    int nc = input.cols;

    // Conversion de l'image en niveaux de gris
    cv::cvtColor(input, input, cv::COLOR_BGR2GRAY);

    // Egalisation de l'histogramme de l'image
    cv::equalizeHist(input, input);

    // Initialisation des centres mk
    float mk[K];
    for (int k = 0; k < K; k++) {
        mk[k] = k * 255 / K + (255 / (2 * K));
    }

    int colors[K];
    // Initialisation des couleurs des K groupes
    colors[0] = 0; 
    colors[K-1] = 255;
    for (int k = 1; k < K - 1; k++) {
        colors[k] = (int) ((k * 255) / (K-1));
    }

    // Initialisation des nouveaux centres mk et du nombre d'éléments par groupe
    float mkNew[K];
    int nbElements[K];
    for (int k = 0; k < K; k++) {
        mkNew[k] = 0.0;
        nbElements[k] = 0;
    }

    std::vector<int> distances;
    int diff;
    std::vector<int>::iterator result;
    int minIndex;
    cv::Mat output(nl, nc, CV_8UC1);

    while (distanceMeans(mk, mkNew, K) > eps) {
        // Mise à jour des moyennes
        if (mkNew[K-1] != 0.0) {
            for (int k = 0; k < K; k++) {
                mk[k] = mkNew[k];
                mkNew[k] = 0;
            }
        }

        // Calcul des nouveaux groupes
        for (int i = 0; i < nl; i++) {
            for (int j = 0; j < nc; j++) {
                for (int k = 0; k < K; k++) {
                    diff = (int)input.at<uchar>(i,j) - mk[k];
                    distances.push_back(sqrt(diff * diff)); 
                }
                result = std::min_element(distances.begin(), distances.end());
                minIndex = std::distance(distances.begin(), result);
                output.at<uchar>(i,j) = minIndex;
                distances.clear();
            }
        }

        // Calcul de la nouvelle moyenne de chaque groupe
        for (int i = 0; i < nl; i++) {
            for (int j = 0; j < nc; j++) {
                int indice = (int)output.at<uchar>(i,j); 
                mkNew[indice] += input.at<uchar>(i,j);
                nbElements[indice]++;
            }
        }
        for (int k = 0; k < K; k++) {
           mkNew[k] /= nbElements[k];
           nbElements[k] = 0;
        }
    }

    // Affectation des couleurs de chaque classe
    for (int i = 0; i < nl; i++) {
        for (int j = 0; j < nc; j++) {
            output.at<uchar>(i,j) = colors[(int)output.at<uchar>(i,j)];
        }
    }

    return output;
}
