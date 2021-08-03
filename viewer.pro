QT += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = main
TEMPLATE = app

DEFINES += QT_DEPRECATED_WARNINGS

SOURCES += \
    src/detection.cpp \
    src/filter.cpp \
	src/kernel.cpp \
    src/mainWindow.cpp \
 	src/openclProgram.cpp \
    src/outils.cpp \
    src/popup.cpp \
    src/segmentation.cpp \
    src/violaJones.cpp \
    main.cpp

INCLUDEPATH += \
    include \
    3rdparty

HEADERS += \
    include/detection.h \
    include/filter.h \
	include/kernel.h \
    include/mainWindow.h \
	include/openclProgram.h \
    include/outils.h \
    include/popup.h \
    include/segmentation.h \
    include/violaJones.h \
    3rdparty/stb_image.h \
    3rdparty/stb_image_write.h

CONFIG += link_pkgconfig
PKGCONFIG += opencv

LIBS += \
   `pkg-config opencv --cflags --libs` \
   -lOpenCL
