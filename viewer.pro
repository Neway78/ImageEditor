QT += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = main
TEMPLATE = app

DEFINES += QT_DEPRECATED_WARNINGS

SOURCES += \
    src/detection.cpp \
    src/filter.cpp \
    src/mainWindow.cpp \
    src/outils.cpp \
    src/popup.cpp \
    src/segmentation.cpp \
    src/violaJones.cpp \
    main.cpp

HEADERS += \
    include/detection.h \
    include/filter.h \
    include/mainWindow.h \
    include/outils.h \
    include/popup.h \
    include/segmentation.h \
    include/violaJones.h

INCLUDEPATH += \
    include

CONFIG += link_pkgconfig
PKGCONFIG += opencv

LIBS += \
   `pkg-config opencv --cflags --libs` \
