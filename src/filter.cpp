#include "filter.h"

double *gaussianKernel;
double normalizeCoeff;

static float filter5x5[25] = {1.0/256,  4.0/256,  6.0/256,  4.0/256, 1.0/256, 
                              4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256,
                              6.0/256, 24.0/256, 36.0/256, 24.0/256, 6.0/256, 
                              4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256,
                              1.0/256,  4.0/256,  6.0/256,  4.0/256, 1.0/256};

// Convolution pour un pixel donné
uchar convolutionPixel(const cv::Mat &input, const int halfMaskSize, const double sigma, const int x, const int y)
{
    int nl = input.rows; 
    int nc = input.cols;

    double result = 0;
    double tmpSum = 0;
    for (int i = -halfMaskSize; i <= halfMaskSize; i++) {
        for (int j = - halfMaskSize; j <= halfMaskSize; j++) {
            tmpSum += gaussianKernel[j+halfMaskSize] * (double)input.at<uchar>((x+i+nl)%nl, (y+j+nc)%nc);
        }
        result +=tmpSum * gaussianKernel[i+halfMaskSize];
        tmpSum = 0;
    }
    uchar coeff = (uchar)result;
    return coeff;
}

// Filtrage gaussien par convolution spatiale
cv::Mat gaussConvolutionFilter(cv::Mat input, const int maskSize, const double sigma) 
{
    printf(">> Spatial Gaussian Blur: maskSize = %i; sigma = %f\n\n", maskSize, sigma);

    // Conversion de l'image en niveau de gris
    cv::cvtColor(input, input, cv::COLOR_BGR2GRAY);

    if (sigma == 0.0) {
        return input;
    }

    // Initialisation du Gauss Kernel
    const int halfMaskSize = maskSize/2;
    gaussianKernel = new double[maskSize];

    // Calcul du Gauss Kernel
    const double sigma2 = 2 * sigma * sigma;
    double normalizeCoeff = 0;
    for (int i = -halfMaskSize; i <= halfMaskSize; i++) {
        gaussianKernel[i+halfMaskSize] = (exp(-(i*i) / sigma2)) / (PI * sigma2);
        normalizeCoeff +=  gaussianKernel[i+halfMaskSize];
    }

    for (int i = 0; i < maskSize; i++) {
        gaussianKernel[i] /= normalizeCoeff;
    }

    cv::Mat cvOutput(input.rows, input.cols, CV_8UC1);

    for (int x = 0; x < input.rows; x++) {
        for (int y = 0; y < input.cols; y++) {
            cvOutput.at<uchar>(x,y) = convolutionPixel(input, halfMaskSize, sigma, x, y);
        }
    }
    delete gaussianKernel;

    return cvOutput;
}

// Convolution pour un pixel donné
float convolutionPixelGrad(const cv::Mat &input, const int halfMaskSize, const double sigma, const int x, const int y)
{
    int nl = input.rows; 
    int nc = input.cols;

    double result = 0;
    double tmpSum = 0;
    for (int i = -halfMaskSize; i <= halfMaskSize; i++) {
        for (int j = - halfMaskSize; j <= halfMaskSize; j++) {
            tmpSum += gaussianKernel[j+halfMaskSize] * (double)input.at<float>((x+i+nl)%nl, (y+j+nc)%nc);
        }
        result +=tmpSum * gaussianKernel[i+halfMaskSize];
        tmpSum = 0;
    }
    float coeff = (float)result;
    return coeff;
}

// Filtrage gaussien par convolution spatiale
cv::Mat gaussConvolutionFilterGrad(cv::Mat input, const int maskSize, const double sigma) 
{
    // Initialisation du Gauss Kernel
    const int halfMaskSize = maskSize/2;
    gaussianKernel = new double[maskSize];

    // Calcul du Gauss Kernel
    const double sigma2 = 2 * sigma * sigma;
    double normalizeCoeff = 0;
    for (int i = -halfMaskSize; i <= halfMaskSize; i++) {
        gaussianKernel[i+halfMaskSize] = (exp(-(i*i) / sigma2)) / (PI * sigma2);
        normalizeCoeff +=  gaussianKernel[i+halfMaskSize];
    }

    for (int i = 0; i < maskSize; i++) {
        gaussianKernel[i] /= normalizeCoeff;
    }

    cv::Mat cvOutput(input.rows, input.cols, CV_32FC1);

    for (int x = 0; x < input.rows; x++) {
        for (int y = 0; y < input.cols; y++) {
            cvOutput.at<float>(x,y) = convolutionPixelGrad(input, halfMaskSize, sigma, x, y);
        }
    }
    delete gaussianKernel;

    return cvOutput;
}

// Echange des quadrants pour la fft
void fftshift(cv::Mat fourierMat)
{
    int cx = fourierMat.cols/2;
    int cy = fourierMat.rows/2;

    cv::Mat tmp;
    cv::Mat q0(fourierMat, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(fourierMat, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(fourierMat, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(fourierMat, cv::Rect(cx, cy, cx, cy));

    // Echange des quadrants 1 et 4
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    // Echange des quadrants 2 et 3
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

// Convolution par une gausienne pour la fft
double fftConvolutionPixel(const double sigma, int u, int v, int N, int M)
{
    assert(0 <= u && u < N && 0 <= v && v < M);

    return exp(-2 * PI_SQUARE * pow(sigma, 2) 
               * (  pow(((u - (double) N / 2) / N), 2)
                  + pow(((v - (double) M / 2) / M), 2)));
}

// Filtrage gaussien par fft
cv::Mat fftConvolution(cv::Mat input, const double sigma)
{
    printf(">> Time Gaussian Blur: sigma = %f\n\n", sigma);

    // Dimensions de l'image
    int nl = input.rows;
    int nc = input.cols;

    // Conversion de l'image en niveau de gris
    cv::cvtColor(input, input, cv::COLOR_BGR2GRAY);

    // Conversion en image de réels simple précision
    cv::Mat fImage;
    input.convertTo(fImage, CV_64F);

    // Algorithme Discrete Fourier Transform
    cv::Mat fourierTransform;
    dft(fImage, fourierTransform, cv::DFT_COMPLEX_OUTPUT);

    // Echange des quadrants de la FFT
    fftshift(fourierTransform);

    // Convolution pour chaque pixel
    double coef_convolution;
    for (int i = 0; i < nl; i++) {
        for (int j = 0; j < nc; j++) {
            coef_convolution = fftConvolutionPixel(sigma, i, j, nl, nc);
            fourierTransform.at<std::complex<double>>(i,j) = 
                coef_convolution * fourierTransform.at<std::complex<double>>(i,j);
        }
    }

    // Echange des quadrants avant de faire la FFT-1
    fftshift(fourierTransform);

    // Image originale
    cv::Mat inverseTransform;
    dft(fourierTransform, inverseTransform, cv::DFT_SCALE | cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);

    // Conversion en image de uchar
    cv::Mat output;
    inverseTransform.convertTo(output, CV_8UC1);
    return output;
}

// Laplacien de l'image
cv::Mat laplacianGPUFilter(unsigned char *input, int width, int height, OpenCLProgram &ourProgram)
{
    const char *kernelPath = "../kernels/gaussian.cl";
    const char *kernelName = "gaussian";
    Kernel gaussianGPUKernel(ourProgram._context, ourProgram._devices,
                             kernelPath, kernelName);

    cl::ImageFormat imageFormat = cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8);
    cl::Buffer filter = cl::Buffer(ourProgram._context, CL_MEM_READ_ONLY, 25*sizeof(float));
    cl::Sampler sampler = cl::Sampler(ourProgram._context, CL_FALSE, 
                                      CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST);

    cl::Image2D inputImage;
    cl::Image2D outputImage;
    cl::size_t<3> origin;
    cl::size_t<3> region;

    try
    {
        inputImage = cl::Image2D(ourProgram._context, CL_MEM_READ_ONLY, 
                                 imageFormat, width, height);
        outputImage = cl::Image2D(ourProgram._context, CL_MEM_READ_WRITE,
                                  imageFormat, width, height);

        origin[0] = 0; origin[1] = 0; origin[2] = 0;
        region[0] = width; region[1] = height; region[2] = 1;

        ourProgram._queue.enqueueWriteImage(inputImage, CL_TRUE, origin, region, 0, 0, input);
        ourProgram._queue.enqueueWriteBuffer(filter, CL_TRUE, 0, 25*sizeof(float), filter5x5);	

        gaussianGPUKernel._kernel.setArg(0, inputImage);
        gaussianGPUKernel._kernel.setArg(1, outputImage);
        gaussianGPUKernel._kernel.setArg(2, filter);
        gaussianGPUKernel._kernel.setArg(3, sampler);

        cl::NDRange global(width, height);
        cl::NDRange local(8,8);
        ourProgram._queue.enqueueNDRangeKernel(gaussianGPUKernel._kernel, cl::NullRange, global, local);
    } 
    catch (cl::Error &error) 
    {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    kernelPath = "../kernels/laplacian.cl";
    kernelName = "laplacian";
    Kernel laplacianGPUKernel(ourProgram._context, ourProgram._devices,
                              kernelPath, kernelName);

    unsigned char * outputGPU = (unsigned char *)malloc(sizeof(unsigned char) * width * height * 4);

    try
    {
        cl::Image2D gaussianDown = inputImage;
        cl::Image2D gaussianUp = outputImage;
        cl::Image2D outputLaplacian = cl::Image2D(ourProgram._context, CL_MEM_WRITE_ONLY,
                                                  imageFormat, width, height);

        laplacianGPUKernel._kernel.setArg(0, gaussianDown);
        laplacianGPUKernel._kernel.setArg(1, gaussianUp);
        laplacianGPUKernel._kernel.setArg(2, outputLaplacian);
        laplacianGPUKernel._kernel.setArg(3, sampler);

        cl::NDRange global(width, height);
        cl::NDRange local(8, 8);
        ourProgram._queue.enqueueNDRangeKernel(laplacianGPUKernel._kernel, cl::NullRange, global, local);

        ourProgram._queue.enqueueReadImage(outputLaplacian, CL_TRUE, origin, region, 0, 0, outputGPU);

    }
    catch (cl::Error &error) 
    {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::Mat output(height, width, CV_8UC4);
    std::memcpy(output.data, outputGPU, sizeof(unsigned char) * width * height * 4);

    free(outputGPU);

    return output;
}

// Filtrage médian
cv::Mat computeMedian(cv::Mat input, const int size) 
{
    printf(">> Median Blur: maskSize = %i\n\n", size);

    // dimensions de l'image 
    int nl = input.rows;
    int nc = input.cols;

    // Conversion de l'image en niveau de gris
    cv::cvtColor(input, input, cv::COLOR_BGR2GRAY);

    cv::Mat output(nl, nc, CV_8UC1);

    // taille de la fenêtre du filtre
    int demiFiltre = size/2;
    int tailleFiltre = size * size;

    // position de la médiane
    int positionMediane = tailleFiltre/2 + 1;

    // initialisation de l'histogramme cumulé
    int histCumule[256 * sizeof(int)];

    // valeur du pixel de l'image
    uchar valPixel;

    // Parcours en ligne de l'image initiale
    for (int l = 0; l < nl; l++) {

        // on applique la fenêtre du filtre sur la première colonne (indice 0)
        for (int i = -demiFiltre; i <= demiFiltre; i++) {
            for (int j = -demiFiltre; j <= demiFiltre; j++) {
                valPixel = input.at<uchar>((l+i+nl)%nl,(j+nc)%nc);
                for (int v = valPixel; v < 256; v++) {
                    histCumule[v] += 1;
                }
            }
        }

        // Parcours de l'histogramme cumulé pour trouver la médiane
        int indice = 0;
        while (histCumule[indice] < positionMediane) {
            indice ++;
        }
        output.at<uchar>(l,0) = indice;
        
        // Pour les colonnes suivantes, on actualise l'histogramme
        for (int c = 1; c < nc; c++) {
            for (int i = -demiFiltre; i<= demiFiltre; i++) {

                // On supprime la première colonne
                int iIndex = (l+i+nl)%nl;
                int jIndex = (c-demiFiltre-1+nc)%nc;
                valPixel = input.at<uchar>(iIndex,jIndex);
                for (int v = valPixel; v < 256; v++) {
                    histCumule[v] -= 1;
                }

                // On ajoute ensuite celle qui suit la dernière
                iIndex = (l+i+nl)%nl;
                jIndex = (c+demiFiltre+nc)%nc;
                valPixel = input.at<uchar>(iIndex,jIndex);
                for (int v = valPixel; v < 256; v++) {
                    histCumule[v] += 1;
                }
            }

            // Parcours de l'histogramme cumulé pour trouver la médiane
            indice = 0;
            while (histCumule[indice] < positionMediane) {
                indice ++;
            }
            output.at<uchar>(l,c) = indice;
        }
        // Fin du parcours de la ligne : on remet l'histogramme à zéro
        for (int v = 0; v < 256; v++) {
            histCumule[v] = 0;
        }
    }
    return output;
}

// Filtrage bilatéral
cv::Mat computeBilateral(cv::Mat input, const double sigma1, 
                         const double sigma2, const int nbIter)
{
    printf(">> Bilateral Filter: sigma1 = %f; sigma2 = %f; nbIter = %i\n\n", sigma1, sigma2, nbIter);

    int nl = input.rows;
    int nc = input.cols;

    // Conversion de l'image en niveau de gris
    cv::cvtColor(input, input, cv::COLOR_BGR2GRAY);

    cv::Mat output(nl, nc, CV_8UC1);

    double numerateur, denominateur; 
    double pixelValue;
    double coeffExp;

    int sigma1Int = (int) sigma1;

    for (int t = 1; t <= nbIter; t++) {
        for (int x = 0; x < nl; x++) {
            for (int y = 0; y < nc; y++) {
                numerateur = 0;
                denominateur = 0;
                for (int i = -3*sigma1Int; i <= 3*sigma1Int; i++) {
                    for (int j = -3*sigma1Int; j <= 3*sigma1Int; j++) {
                        pixelValue = input.at<uchar>((x+i+nl)%nl,(y+j+nc)%nc);

                        coeffExp = exp(-(i*i + j*j) / (2 * sigma1 * sigma1)
                                       - (pow(pixelValue - (double)input.at<uchar>(x,y),2))
                                          / (2 * sigma2 * sigma2)
                                      );
                        numerateur += coeffExp * pixelValue;
                        denominateur += coeffExp;
                    }
                }
                output.at<uchar>(x,y) = (uchar)(numerateur / denominateur);
            }
        }
        input = output;
    }
    return output;
}
