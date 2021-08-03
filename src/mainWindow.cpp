#include "mainWindow.h"
#include "popup.h"
#include "filter.h"
#include "detection.h"
#include "segmentation.h"
#include "violaJones.h"
#include "outils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , inputImageLabel(new QLabel())
    , outputImageLabel(new QLabel())
    , initializeOpenCL(false)
{
    createMenus();
    
    // Sous-Fenêtres
    inputImageLabel->setStyleSheet("QLabel { background-color : gray }");
    outputImageLabel->setStyleSheet("QLabel { background-color : gray }");
    QHBoxLayout *innerWindowLayout = new QHBoxLayout();
    innerWindowLayout->addWidget(inputImageLabel);
    innerWindowLayout->addSpacing(3);
    innerWindowLayout->addWidget(outputImageLabel);

    QWidget *innerWindow = new QWidget();
    innerWindow->setLayout(innerWindowLayout);
    setCentralWidget(innerWindow);

    setWindowTitle("Image Processing Using QT and OpenCV");
    setMinimumSize(500,250);
    resize(1000,400);

    std::cout << "\n--------------------------------------------------------\n";
    std::cout << "          Image Processing Using QT and OpenCV          \n";
    std::cout << "--------------------------------------------------------\n\n";
}

void MainWindow::createMenus()
{
    fileMenu = menuBar()->addMenu(tr("&File"));
    QAction *openImg = fileMenu->addAction(tr("&Open Image"), this, &MainWindow::open);
    openImg->setShortcut(QKeySequence::Open);
    QAction *closeImg = fileMenu->addAction(tr("&Close Image"), this, &MainWindow::close);
    closeImg->setShortcut(tr("Ctrl+Q"));
    fileMenu->addAction("Save");
    fileMenu->addSeparator();
    QAction *closeViewer = fileMenu->addAction(tr("&Exit"), this, &QWidget::close);
    closeViewer->setShortcut(QKeySequence::Close);

    filteringMenu = menuBar()->addMenu(tr("&Filtering"));
    gaussianSubMenu = filteringMenu->addMenu(tr("&Gaussian Filter"));
    gaussianSubMenu->addAction(tr("&Spatial Domain"), this, &MainWindow::gaussFilterSpatial);
    gaussianSubMenu->addAction(tr("&Time Domain"), this, &MainWindow::gaussFilterTime);
	filteringMenu->addAction(tr("&Laplacian Filter"), this, &MainWindow::laplacianFilter);
    filteringMenu->addAction(tr("&Median Filter"), this, &MainWindow::medianFilter);
    filteringMenu->addAction(tr("&Bilateral Filter"), this, &MainWindow::bilateralFilter);

    detectionMenu = menuBar()->addMenu(tr("&Feature detection"));
    EdgesSubMenu = detectionMenu->addMenu(tr("&Edges"));
    EdgesSubMenu->addAction(tr("&Canny"), this, &MainWindow::cannyEdgeDetection);
    PointsSubMenu = detectionMenu->addMenu(tr("&Interest points"));
    PointsSubMenu->addAction(tr("&Moravec detector"), this, &MainWindow::moravecDetector);
    PointsSubMenu->addAction(tr("&Harris detector"), this, &MainWindow::harrisDetector);
    ShapesSubMenu = detectionMenu->addMenu(tr("&Shapes"));
    ShapesSubMenu->addAction(tr("&Hough Transform - Lines"), this, &MainWindow::houghTransform);

    segmentationMenu = menuBar()->addMenu(tr("&Segmentation"));
    segmentationMenu->addAction(tr("&Ostu's Method"), this, &MainWindow::segOtsu);
    segmentationMenu->addAction(tr("&K-Means"), this, &MainWindow::kMeans);

    AIMenu = menuBar()->addMenu(tr("&Machine Learning"));
    AIMenu->addAction(tr("&Viola Jones Face Detection"), this, &MainWindow::violaJones);
}

void MainWindow::open()
{
    imagePath = QFileDialog::getOpenFileName(this, tr("Open Image"), QDir::currentPath(),tr("Images (*.png *.jpg *.jpeg)"));
    if (imagePath != "") {
        std::cout << ">> Loading Image: " << imagePath.toStdString() << "\n\n";
        inputImage = QImage(imagePath);
        inputImageLabel->setPixmap(QPixmap::fromImage(inputImage.scaled(inputImageLabel->size(), Qt::KeepAspectRatio)));
    }

    if (!outputImage.isNull()) {
        outputImage = QImage();
    }
    outputImageLabel->setPixmap(QPixmap());
}

void MainWindow::close()
{
    if (!inputImage.isNull()) {
        inputImage = QImage();
    }
    inputImageLabel->setPixmap(QPixmap());

    if (!outputImage.isNull()) {
        outputImage = QImage();
    }
    outputImageLabel->setPixmap(QPixmap());
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    if (!inputImage.isNull()) {
        inputImageLabel->setPixmap(QPixmap::fromImage(inputImage.scaled(inputImageLabel->size(), Qt::KeepAspectRatio)));
    }
    if (!outputImage.isNull()) {
        outputImageLabel->setPixmap(QPixmap::fromImage(outputImage.scaled(outputImageLabel->size(), Qt::KeepAspectRatio)));
    }
}

void MainWindow::retrieveFields(QStringList outputFields)
{
    if (!paramFields.isEmpty()) {
        paramFields.clear();
    }
    paramFields = outputFields;
}

void MainWindow::gaussFilterSpatial()
{
    if (!inputImage.isNull()) {
        cv::Mat cvInputImage = cv::imread(imagePath.toStdString());
        
        QStringList filterFields = {"Taille du Kernel:", "Sigma:"};
        QStringList valuesFields = {"5", "2.0"};
        PopUp *filterParams = new PopUp(&filterFields, &valuesFields, this);
        filterParams->exec();

        bool ok;
        const int maskSize = paramFields.at(0).toInt(&ok, 10); 
        const double sigma = paramFields.at(1).toDouble(&ok);

        cv::Mat_<uchar>& cvInputImageTemp = (cv::Mat_<uchar>&)cvInputImage;
        cv::Mat cvOutputImage = gaussConvolutionFilter(cvInputImageTemp, maskSize, sigma);

        outputImage = Mat2QImage(cvOutputImage);
        outputImageLabel->setPixmap(QPixmap::fromImage(outputImage.scaled(outputImageLabel->size(), Qt::KeepAspectRatio)));
        paramFields.clear();
    }
}

void MainWindow::gaussFilterTime()
{
    if (!inputImage.isNull()) {
        cv::Mat cvInputImage = cv::imread(imagePath.toStdString());
        
        QStringList filterFields = {"Sigma:"};
        QStringList valuesFields = {"2.0"};
        PopUp *filterParams = new PopUp(&filterFields, &valuesFields, this);
        filterParams->exec();

        bool ok;
        const double sigma = paramFields.at(0).toDouble(&ok);     

        cv::Mat cvOutputImage = fftConvolution(cvInputImage, sigma);
        outputImage = Mat2QImage(cvOutputImage);
        outputImageLabel->setPixmap(QPixmap::fromImage(outputImage.scaled(outputImageLabel->size(), Qt::KeepAspectRatio)));
        paramFields.clear();
    }
}

void MainWindow::laplacianFilter()
{
    if (!inputImage.isNull()) {
        int width, height, channels;
        unsigned char *inputImage = stbi_load(imagePath.toStdString().c_str(), &width, &height, &channels, 4);

        if (width % 8 == 0 && height % 8 == 0) {
            if (!initializeOpenCL) {
                ourProgram.initProgram();
                initializeOpenCL = true;
            }

            cv::Mat cvOutputImage = laplacianGPUFilter(inputImage, width, height, ourProgram); 
            outputImage = Mat2QImage(cvOutputImage);
            outputImageLabel->setPixmap(QPixmap::fromImage(outputImage.scaled(outputImageLabel->size(), Qt::KeepAspectRatio)));
        } else {
            std::cout << "Unable to perform laplacian: Image size must be divisible by 8" << std::endl;
        }
    }
}

void MainWindow::medianFilter()
{
     if (!inputImage.isNull()) {
        cv::Mat cvInputImage = cv::imread(imagePath.toStdString());
        
        QStringList filterFields = {"Size:"};
        QStringList valuesFields = {"5"};
        PopUp *filterParams = new PopUp(&filterFields, &valuesFields, this);
        filterParams->exec();

        bool ok;
        const int size = paramFields.at(0).toInt(&ok);    

        cv::Mat cvOutputImage = computeMedian(cvInputImage, size);
        outputImage = Mat2QImage(cvOutputImage);
        outputImageLabel->setPixmap(QPixmap::fromImage(outputImage.scaled(outputImageLabel->size(), Qt::KeepAspectRatio)));
        paramFields.clear();
    }
}

void MainWindow::bilateralFilter()
{
     if (!inputImage.isNull()) {
        cv::Mat cvInputImage = cv::imread(imagePath.toStdString());
        
        QStringList filterFields = {"Sigma1:", "Sigma2:", "NbIter:"};
        QStringList valuesFields = {"4.0", "40.0", "1"};
        PopUp *filterParams = new PopUp(&filterFields, &valuesFields, this);
        filterParams->exec();

        bool ok;
        const double sigma1 = paramFields.at(0).toDouble(&ok);       
        const double sigma2 = paramFields.at(1).toDouble(&ok);       
        const int nbIter = paramFields.at(2).toInt(&ok);

        cv::Mat cvOutputImage = computeBilateral(cvInputImage, sigma1, sigma2, nbIter);
        outputImage = Mat2QImage(cvOutputImage);
        outputImageLabel->setPixmap(QPixmap::fromImage(outputImage.scaled(outputImageLabel->size(), Qt::KeepAspectRatio)));
        paramFields.clear();
    }
}

void MainWindow::cannyEdgeDetection()
{
    if (!inputImage.isNull()) { 
        cv::Mat cvInputImage = cv::imread(imagePath.toStdString());

        QStringList filterFields = {"maskSize:", "Sigma:", "Low Threshold:", "High Threshold"};
        QStringList valuesFields = {"5", "0.4", "100", "150"};
        PopUp *filterParams = new PopUp(&filterFields, &valuesFields, this);
        filterParams->exec();

        bool ok;
        const int maskSize = paramFields.at(0).toInt(&ok);       
        const double sigma = paramFields.at(1).toDouble(&ok); 
        const int seuilBas = paramFields.at(2).toInt(&ok);
        const int seuilHaut = paramFields.at(3).toInt(&ok);

        cv::Mat cvOutputImage = computeCanny(cvInputImage, maskSize, sigma, seuilBas, seuilHaut);
        outputImage = Mat2QImage(cvOutputImage);        
        outputImageLabel->setPixmap(QPixmap::fromImage(outputImage.scaled(outputImageLabel->size(), Qt::KeepAspectRatio)));
    }
}

void MainWindow::moravecDetector()
{
    if (!inputImage.isNull()) { 
        cv::Mat cvInputImage = cv::imread(imagePath.toStdString());

        QStringList filterFields = {"Threshold:"};
        QStringList valuesFields = {"180"};
        PopUp *filterParams = new PopUp(&filterFields, &valuesFields, this);
        filterParams->exec();

        bool ok;
        const int threshold = paramFields.at(0).toInt(&ok);       

        cv::Mat cvOutputImage = computeMoravec(cvInputImage, threshold);
        outputImage = Mat2QImage(cvOutputImage);        
        outputImageLabel->setPixmap(QPixmap::fromImage(outputImage.scaled(outputImageLabel->size(), Qt::KeepAspectRatio)));
    }
}

void MainWindow::harrisDetector()
{
    if (!inputImage.isNull()) { 
        cv::Mat cvInputImage = cv::imread(imagePath.toStdString());

        QStringList filterFields = {"[Gaussian Filter] - Mask Size:", "[Gaussian Filter] - Sigma:", "Threshold - R:"};
        QStringList valuesFields = {"5", "2.0", "1000000.0"};
        PopUp *filterParams = new PopUp(&filterFields, &valuesFields, this);
        filterParams->exec();

        bool ok;
        const int maskSize = paramFields.at(0).toInt(&ok);
        const double sigma = paramFields.at(1).toDouble(&ok);
        const float R = paramFields.at(2).toFloat(&ok);     

        cv::Mat cvOutputImage = computeHarris(cvInputImage, maskSize, sigma, R);
        outputImage = Mat2QImage(cvOutputImage);        
        outputImageLabel->setPixmap(QPixmap::fromImage(outputImage.scaled(outputImageLabel->size(), Qt::KeepAspectRatio)));
    }
}

void MainWindow::houghTransform()
{
    if (!inputImage.isNull()) { 
        cv::Mat cvInputImage = cv::imread(imagePath.toStdString());

        QStringList filterFields = {"[Canny] maskSize:", "[Canny] Sigma:", "[Canny] Low Threshold:", "[Canny] High Threshold", "Threshold:"};
        QStringList valuesFields = {"5", "2.0", "100", "150", "100"};
        PopUp *filterParams = new PopUp(&filterFields, &valuesFields, this);
        filterParams->exec();

        bool ok;
        const int maskSize = paramFields.at(0).toInt(&ok);
        const double sigma = paramFields.at(1).toDouble(&ok);
        const int seuilBas = paramFields.at(2).toInt(&ok);
        const int seuilHaut = paramFields.at(3).toInt(&ok);
        const int threshold = paramFields.at(4).toInt(&ok);    

        cv::Mat cvOutputImage = computeHough(cvInputImage, maskSize, sigma, seuilBas, seuilHaut, threshold);
        outputImage = Mat2QImage(cvOutputImage);        
        outputImageLabel->setPixmap(QPixmap::fromImage(outputImage.scaled(outputImageLabel->size(), Qt::KeepAspectRatio)));
    }
}

void MainWindow::segOtsu()
{
    if (!inputImage.isNull()) {
        cv::Mat cvInputImage = cv::imread(imagePath.toStdString());
        cv::Mat cvOutputImage = computeOtsu(cvInputImage);
        outputImage = Mat2QImage(cvOutputImage);        
        outputImageLabel->setPixmap(QPixmap::fromImage(outputImage.scaled(outputImageLabel->size(), Qt::KeepAspectRatio)));
    }
}

void MainWindow::kMeans()
{
    if (!inputImage.isNull()) { 
        cv::Mat cvInputImage = cv::imread(imagePath.toStdString());

        QStringList filterFields = {"Number Groups - K:", "Epsilon"};
        QStringList valuesFields = {"2", "0.1"};
        PopUp *filterParams = new PopUp(&filterFields, &valuesFields, this);
        filterParams->exec();

        bool ok;
        const int K = paramFields.at(0).toInt(&ok);
        const float epsilon = paramFields.at(1).toFloat(&ok);   

        cv::Mat cvOutputImage = computeKMeans(cvInputImage, K, epsilon);
        outputImage = Mat2QImage(cvOutputImage);        
        outputImageLabel->setPixmap(QPixmap::fromImage(outputImage.scaled(outputImageLabel->size(), Qt::KeepAspectRatio)));
    }
}

void MainWindow::violaJones()
{
    if (!inputImage.isNull()) { 
        cv::Mat cvInputImage = cv::imread(imagePath.toStdString());
        cv::Mat cvOutputImage = haarCascadeClassifier(cvInputImage);
        outputImage = Mat2QImage(cvOutputImage);
        outputImageLabel->setPixmap(QPixmap::fromImage(outputImage.scaled(outputImageLabel->size(), Qt::KeepAspectRatio)));
    }
}
