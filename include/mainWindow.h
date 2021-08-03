#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QtWidgets>

#include "openclProgram.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    bool loadFile(const QString &);

private slots:
    void open();
    void close();
    void resizeEvent(QResizeEvent *);
    void retrieveFields(QStringList outputFields);
    void gaussFilterSpatial();
    void gaussFilterTime();
	void laplacianFilter();
    void medianFilter();
    void bilateralFilter();
    void cannyEdgeDetection();
    void moravecDetector();
    void harrisDetector();
    void houghTransform();
    void segOtsu();
    void kMeans();
    void violaJones();
    
private:
    void createMenus();

    QMenu *fileMenu;
    QMenu *filteringMenu;
    QMenu *gaussianSubMenu;
    QMenu *detectionMenu;
    QMenu *EdgesSubMenu;
    QMenu *PointsSubMenu;
    QMenu *ShapesSubMenu;
    QMenu *segmentationMenu;
    QMenu *AIMenu;

    QString imagePath;
    QImage inputImage;
    QImage outputImage;
    QLabel *inputImageLabel;
    QLabel *outputImageLabel;
    QStringList paramFields;

    bool initializeOpenCL;
    OpenCLProgram ourProgram;   
};

#endif
