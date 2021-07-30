#include <QApplication>
#include <QtWidgets>

#include "mainWindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    MainWindow interface;
    interface.show();
    
    return app.exec();
}