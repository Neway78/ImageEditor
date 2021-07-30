#include "detection.h"
#include "filter.h"

// Calcul des composantes Gx et Gy du gradient
void convolutionPixelGradient(const cv::Mat &input, cv::Mat &Gx, cv::Mat &Gy, 
                              int x, int y, int nl, int nc, 
                              int masqueX[3][3], int masqueY[3][3])
{
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            Gx.at<short>(x,y) += masqueX[i+1][j+1] * input.at<uchar>((x-i+nl)%nl, (y-j+nc)%nc);
            Gy.at<short>(x,y) += masqueY[i+1][j+1] * input.at<uchar>((x-i+nl)%nl, (y-j+nc)%nc);
        }
    }
}

// Calcul du gradient aux points P1 et P2
void computeGradientPoints(const cv::Mat &G, const cv::Mat &Gx, const cv::Mat &Gy,
                           cv::Mat &gradP1, cv::Mat &gradP2, int nl, int nc) 
{
    short gradX, gradY;
    float interp1, interp2;
    for (int i = 0; i < nl; i++) {
        for (int j = 0; j < nc; j++) {
            gradX = Gx.at<short>(i,j);
            gradY = Gy.at<short>(i,j);
            if (abs(gradX) > abs(gradY)) {
                if ((gradX <= 0 && gradY >= 0) || (gradX >= 0 && gradY <= 0)) {
                    // ]pi/4,pi/2] U ]5pi/4,3pi/2]
                    interp1 = -(float)gradY / gradX;
                    interp2 = (float)(gradX + gradY) / gradX;
                    gradP1.at<float>(i,j) = interp1 * G.at<float>((i-1+nl)%nl,(j+1+nc)%nc) + 
                                            interp2 * G.at<float>((i-1+nl)%nl,j);
                    gradP2.at<float>(i,j) = interp1 * G.at<float>((i+1+nl)%nl,(j-1+nc)%nc) +
                                            interp2 * G.at<float>((i+1+nl)%nl,j);
                } else {
                    interp1 = (float)gradY / gradX;
                    interp2 = (float)(gradX - gradY) / gradX;
                    // ]pi/2,3pi/4] U ]3pi/2,7pi/4[
                    gradP1.at<float>(i,j) = interp1 * G.at<float>((i-1+nl)%nl,(j-1+nc)%nc) +
                                            interp2 * G.at<float>((i-1+nl)%nl,j);
                    gradP2.at<float>(i,j) = interp1 * G.at<float>((i+1+nl)%nl,(j+1+nc)%nc) +
                                            interp2 * G.at<float>((i+1+nl)%nl,j);
                }
            } else {
                if ((gradX <= 0 && gradY >= 0) || (gradX >= 0 && gradY <= 0)) {
                    // [0,pi/4] U [pi,5pi/4]
                    interp1 = -(float)gradX / gradY;
                    interp2 = (float)(gradY + gradX) / gradY;
                    gradP1.at<float>(i,j) = interp1 * G.at<float>((i-1+nl)%nl,(j+1+nc)%nc) +
                                            interp2 * G.at<float>(i,(j+1+nc)%nc);
                    gradP2.at<float>(i,j) = interp1 * G.at<float>((i+1+nl)%nl,(j-1+nc)%nc) +
                                            interp2 * G.at<float>(i,(j-1+nl)%nl);
                } else {
                    // [3pi/4,pi[ U [7pi/4,2pi[
                    interp1 = (float)gradX / gradY;
                    interp2 = (float)(gradY - gradX) / gradY;
                    gradP1.at<float>(i,j) = interp1 * G.at<float>((i+1+nl)%nl,(j+1+nc)%nc) +
                                            interp2 * G.at<float>(i,(j+1+nc)%nc);
                    gradP2.at<float>(i,j) = interp1 * G.at<float>((i-1+nl)%nl,(j-1+nc)%nc) +
                                            interp2 * G.at<float>(i,(j-1+nl)%nl);
                }
            }
        }
    }
}

// Detection des maxima locaux dans la direction du gradient
void detectionMaxima(const cv::Mat &G, const cv::Mat &gradP1, const cv::Mat &gradP2, 
                     cv::Mat &contour, int nl, int nc)
{
    for (int i = 0; i < nl; i++) {
        for (int j = 0; j < nc; j++) {
            if ((G.at<float>(i,j) > abs(gradP1.at<float>(i,j))) && (G.at<float>(i,j) > abs(gradP2.at<float>(i,j)))) {
                contour.at<uchar>(i,j) = 128;
            } else {
                contour.at<uchar>(i,j) = 0;
            }
        }
    }
}

bool checkEdge(const cv::Mat &output, int x, int y, int nl, int nc) {
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if ( i != 0 && j != 0 ) {
                if (output.at<uchar>((x+i+nl)%nl, (y+j+nc)%nc) != 0) {
                    return true;
                }
            }
        }   
    }
    return false;
}

// Extraction des contours 
void gradientExtractContour(const cv::Mat &G, cv::Mat &contour, int seuilBas, int seuilHaut, int nl, int nc)
{
    // Extraction des pixels supérieurs au seuil haut
    // Suppression des pixels inférieurs au seuil bas
    for (int i = 0; i < nl; i++) {
        for (int j = 0; j < nc; j++) {
            if (contour.at<uchar>(i,j) == 128 && G.at<float>(i,j) > seuilHaut) {
                contour.at<uchar>(i,j) = 255;
            } else if (contour.at<uchar>(i,j) == 128 && G.at<float>(i,j) < seuilBas) {
                contour.at<uchar>(i,j) = 0;
            }
        }
    }

    // Extraction des pixels compris entre seuil bas et seuil haut
    for (int i = 0; i < nl; i++) {
        for (int j = 0; j < nc; j++) {
            if (contour.at<uchar>(i,j) == 128) {
                if (checkEdge(contour, i, j, nl, nc)) {
                    contour.at<uchar>(i,j) = 255;
                } else {
                    contour.at<uchar>(i,j) = 0;
                }
            }
        }
    }
}

// Filtre de Canny
cv::Mat computeCanny(cv::Mat input, const int maskSize, const double sigma, 
                     const int seuilBas, const int seuilHaut)
{
    printf(">> Canny Edge Detection: Low Threshold = %i; HighThreshold = %i\n\n", seuilBas, seuilHaut);

    // Dimensions de l'image 
    int nl = input.rows;
    int nc = input.cols;

    // Prewitt :
    // int masqueX[3][3] = {{-1,0,1}, {-1,0,1}, {-1,0,1}};
    // int masqueY[3][3] = {{-1,-1,-1}, {0,0,0}, {1,1,1}};

    // Sobel
    int masqueX[3][3] = {{-1,-2,-1}, {0,0,0}, {1,2,1}};
    int masqueY[3][3] = {{-1,0,1}, {-2,0,2}, {-1,0,1}};

    // Filtrage initial de l'image par convolution spatiale
    cv::Mat filterImage(nl, nc, CV_8UC1);
    filterImage = gaussConvolutionFilter(input, maskSize, sigma);

    // Composantes Gx, Gy et module du gradient 
    cv::Mat Gx(nl, nc, CV_16SC1, cv::Scalar(0));
    cv::Mat Gy(nl, nc, CV_16SC1, cv::Scalar(0));
    cv::Mat G(nl, nc, CV_32FC1);

    // Calcul des composantes Gx et Gy ainsi que du module G
    for (int x = 0; x < nl; x++) {
        for (int y = 0; y < nc; y++) {
            convolutionPixelGradient(filterImage, Gx, Gy, x, y, nl, nc, 
                                     masqueX, masqueY);
            
            // Calcul du module du vecteur gradient
            G.at<float>(x,y) = sqrt((float)(Gx.at<short>(x,y) * Gx.at<short>(x,y) +
                                    Gy.at<short>(x,y) * Gy.at<short>(x,y)));
        }
    }
    
    // Calcul des gradients aux points P1 et P2
    cv::Mat gradP1(nl, nc, CV_32FC1);
    cv::Mat gradP2(nl, nc, CV_32FC1);
    computeGradientPoints(G, Gx, Gy, gradP1, gradP2, nl, nc);

    // Détection des maxima locaux dans la direction du gradient
    cv::Mat contour(nl, nc, CV_8UC1);
    detectionMaxima(G, gradP1, gradP2, contour, nl, nc);

    // Extraction du contour
    gradientExtractContour(G, contour, seuilBas, seuilHaut, nl, nc);

    return contour;
}



// Calcul de la différence d'intensité entre la fenêtre centré en (u,v) 
// et la même fenêtre décalé de (x,y)
float diffIntensiteFenetre(const cv::Mat &input, int nl, int nc, 
                           int u, int v, int halfWindowSize, int x, int y)
{
    float valueWindow = 0;
    int variation;
    for (int i = -halfWindowSize; i <= halfWindowSize; i++) {
        for (int j = -halfWindowSize; j <= halfWindowSize; j++) {
            variation = (int)(input.at<uchar>((u+i+x+nl)%nl, (v+j+y+nc)%nc) -
                              input.at<uchar>((u+i+nl)%nl, (v+j+nc)%nc));
            valueWindow += sqrt(variation * variation);
        }
    }
    return valueWindow;
}

// Détecteur de Moravec
cv::Mat computeMoravec(cv::Mat input, const int threshold)
{
    printf(">> Moravec detector: Threshold = %i\n\n", threshold);

    // Dimensions de l'image
    int nl = input.rows;
    int nc = input.cols;

    // Conversion de l'image en niveau de gris
    cv::cvtColor(input, input, cv::COLOR_BGR2GRAY);

    // Taille de la fenêtre
    int windowSize = 3;
    int halfWindowSize = windowSize / 2;
    int deplacements[2][8] = {{-1,-1,-1,0,0,1,1,1}, {-1,0,1,-1,1,-1,0,-1}};

    cv::Mat ouput(nl, nc, CV_8UC1, cv::Scalar(0));

    // Construction de la carte de coinité
    float valueCarteCoinite;
    float variationIntensite[8];
    int x, y;
    int minVariation = INT_MAX;
    for (int u = 0; u < nl; u++) {
        for (int v = 0; v < nc; v++) {
            for (int k = 0; k < 8; k++) {
                x = deplacements[0][k];
                y = deplacements[1][k];
                variationIntensite[k] = diffIntensiteFenetre(input, nl, nc, u, v, halfWindowSize, x, y);
                if (variationIntensite[k] < minVariation) {
                    minVariation = variationIntensite[k];
                }
            }
            valueCarteCoinite = minVariation; 
            minVariation = INT_MAX;

            if (valueCarteCoinite >= threshold) {
                ouput.at<uchar>(u,v) = 255;
            }
        }
    }

    return ouput;
}



// Détecteur de Harris
cv::Mat computeHarris(cv::Mat input, const int maskSize, const double sigma, const float R)
{
    printf(">> Harris detector: [Gauss] - maskSize = %i, [Gauss] - sigma = %f, Threshold = %f\n", maskSize, sigma, R);

    // Dimensions de l'image
    int nl = input.rows;
    int nc = input.cols;

    // Sobel
    int masqueX[3][3] = {{-1,0,1}, {-2,0,2}, {-1,0,1}};
    int masqueY[3][3] = {{-1,-2,-1}, {0,0,0}, {1,2,1}};

    // Filtrage initial de l'image par convolution spatiale
    cv::Mat filterImage(nl, nc, CV_8UC1);
    filterImage = gaussConvolutionFilter(input, maskSize, sigma);

    // Composantes Gx, Gy et module du gradient 
    cv::Mat Gx(nl, nc, CV_16SC1, cv::Scalar(0));
    cv::Mat Gy(nl, nc, CV_16SC1, cv::Scalar(0));
    cv::Mat G(nl, nc, CV_32FC1);

    // Calcul des composantes Gx et Gy ainsi que du module G
    for (int x = 0; x < nl; x++) {
        for (int y = 0; y < nc; y++) {
            convolutionPixelGradient(filterImage, Gx, Gy, x, y, nl, nc, 
                                     masqueX, masqueY);
            
            // Calcul du module du vecteur gradient
            G.at<float>(x,y) = sqrt(Gx.at<short>(x,y) * Gx.at<short>(x,y) +
                                    Gy.at<short>(x,y) * Gy.at<short>(x,y));
        }
    }
    
    // Calcul des gradients aux points P1 et P2
    cv::Mat gradP1(nl, nc, CV_32FC1);
    cv::Mat gradP2(nl, nc, CV_32FC1);
    computeGradientPoints(G, Gx, Gy, gradP1, gradP2, nl, nc);

    // Calcul de Ix², Iy² et Ixy
    cv::Mat Ix2(nl, nc, CV_32FC1);
    cv::Mat Iy2(nl, nc, CV_32FC1);
    cv::Mat Ixy(nl, nc, CV_32FC1);
    for (int u = 0; u < nl; u++) {
        for (int v = 0; v < nc; v++) {
            Ix2.at<float>(u,v) = gradP1.at<float>(u,v) * gradP1.at<float>(u,v);
            Iy2.at<float>(u,v) = gradP2.at<float>(u,v) * gradP2.at<float>(u,v);
            Ixy.at<float>(u,v) = gradP1.at<float>(u,v) * gradP2.at<float>(u,v);
        }
    }

    // Convolution avec une gaussienne 2D des dérivées
    cv::Mat A(nl, nc, CV_32FC1);
    cv::Mat B(nl, nc, CV_32FC1);
    cv::Mat C(nl, nc, CV_32FC1);
    A = gaussConvolutionFilterGrad(Ix2, maskSize, sigma);
    B = gaussConvolutionFilterGrad(Iy2, maskSize, sigma);
    C = gaussConvolutionFilterGrad(Ixy, maskSize, sigma);

    float valueR[nl][nc];
    float trace;
    for (int u = 0; u < nl; u++) {
        for (int v = 0; v < nc; v++) {
            trace = A.at<float>(u,v) + B.at<float>(u,v);
            valueR[u][v] = (A.at<float>(u,v) * B.at<float>(u,v) - C.at<float>(u,v) * C.at<float>(u,v)) 
                            - 0.04 * trace * trace;
        }
    }

    // Suppression des non maxima
    cv::Mat output(nl, nc, CV_8UC1);
    for (int u = 0; u < nl; u++) {
        for (int v = 0; v < nc; v++) {
            if (valueR[u][v] > R) {
                for (int x = -2; x < 2; x++) {
                    for (int y = -2; y < 2; y++) {
                        if (valueR[u][v] < valueR[(u+x+nl)%nl][(v+y+nc)%nc]) {
                            goto nonMaxima;
                        }
                    }
                }
                
            } else {
                output.at<uchar>(u,v) = 0;
                continue;
            }
            output.at<uchar>(u,v) = 255;
            continue;

            nonMaxima:
            output.at<uchar>(u,v) = 0;
        }
    }

    return output;
}



// Transformée de Hough pour la détection de lignes
cv::Mat computeHough(cv::Mat input, const int maskSize, const double sigma, 
                     const int seuilBas, const int seuilHaut, const int threshold)
{   
    printf(">> Hough Transform: Threshold = %i\n", threshold);

    // Dimensions de l'image
    int nl  = input.rows;
    int nc = input.cols;

    // On commence par faire une filtre de Canny
    cv::Mat output(nl, nc, CV_8UC1);
    output = computeCanny(input, maskSize, sigma, seuilBas, seuilHaut);

    // Création de l'accumulateur dans l'espace de Hough (r,theta)
    double houghH = ((sqrt(2.0) * (double)(nl>nc?nl:nc)) / 2.0);
    int accuH = 2.0 * houghH;
    int accuW = 180;
    int accu[accuH * accuW];

    // Initialisation de l'accumulateur à zéro
    for (int x = 0; x < accuW; x++) {
        for (int y = 0; y < accuH; y++) {
            accu[x * accuH + y] = 0;
        }
    }

    double centerX = nc/2.0;
    double centerY = nl/2.0;

    // On tient compte de toutes les droites passant par le point
    // pour incrémenter l'accumulateur (pas de t = 1 degré)
    for (int y = 0; y < nl; y++) {
        for (int x = 0; x < nc; x++) {
            if (output.at<uchar>(y,x) == 255) {
                for (int t = 0; t < 180; t++) {
                    double r = ((double)x - centerX) * cos((double)t * DEG2RAD) +
                               ((double)y - centerY) * sin((double)t * DEG2RAD);
                    accu[(int)(round(r + houghH) * accuW) + t]++;
                }
            }
        }
    }

    // Création du vecteur de lignes :  [[[x1,y1], [x2,y2]], ...]
    std::vector< std::pair<std::pair<int,int>, std::pair<int,int>> > lines;
    int max;
    int x1, y1, x2, y2;

    // Parcours de l'accumulateur pour trouver les droites supérieurs au seuil
    for (int r = 0; r < accuH; r++) {
        for (int t = 0; t < accuW; t++) {
            if (accu[(r * accuW) + t] >= threshold) {

                // On vérifie qu'il s'agit bien d'un maxima local (voisinage 9*9)
                max = accu[(r * accuW) + t];
                for (int ly = -4; ly <= 4; ly++) {
                    for (int lx = -1; lx <= 4; lx++) {
                        if ((ly+r >= 0 && ly+r < accuH) && (lx+t >= 0 && lx+t < accuW)) {
                            if (accu[(r+ly) * accuW + (t+lx)] >  max) {
                                max = accu[(r+ly) * accuW + (t+lx)];
                                ly = lx = 5;
                            }
                        }
                    }
                }
                if (max > accu[(r*accuW) + t]) {
                    continue;
                }

                // On sait désormais qu'il s'agit d'un maxima local
                // On ajoute donc la droite au vecteur
                x1 = y1 = x2 = y2 = 0;

                if (t >= 45 && t <= 135) {
                    // y = (r - xcos(t)) / sin(t)
                    x1 = 0;
                    y1 = ((double)(r - (accuH/2)) - ((x1 - (nc/2)) * cos(t * DEG2RAD))) 
                            / sin(t * DEG2RAD) + (nl/2);
                    x2 = nc;
                    y2 = ((double)(r - (accuH/2)) - ((x2 - (nc/2)) * cos(t * DEG2RAD))) 
                            / sin(t * DEG2RAD) + (nl/2);  
                } else {
                    // x= (r - ysin(t)) / cos(t)
                    y1 = 0;
                    x1 = ((double)(r - (accuH/2)) - ((y1 - (nl/2)) * sin(t * DEG2RAD))) 
                            / cos(t * DEG2RAD) + (nc/2);
                    y2 = nl;
                    x2 = ((double)(r - (accuH/2)) - ((y2 - (nl/2)) * sin(t * DEG2RAD))) 
                            / cos(t * DEG2RAD) + (nc/2);
                }

                lines.push_back(std::pair<std::pair<int,int>, std::pair<int,int>>(std::pair<int,int>(x1,y1), std::pair<int,int>(x2,y2)));
            }
        }
    }

    // Ajout des lignes détectées sur l'image en sortie
    cv::Mat outputLines(nl, nc, CV_8UC1, cv::Scalar(0));
    std::vector< std::pair< std::pair<int,int>, std::pair<int,int>> >::iterator it;
    for (it = lines.begin(); it != lines.end(); it++) {
        cv::line(outputLines, cv::Point(it->first.first, it->first.second), 
                 cv::Point(it->second.first, it->second.second), cv::Scalar(255), 2, 8);
    }

    return outputLines;
}

