cmake_minimum_required(VERSION 3.1)
project(projetQtTi)

# add flags for compilation
set(CMAKE_CXX_FLAGS "-g -Wall -fPIC")

# flag for debug
set(CMAKE_BUILD_TYPE DEBUG)

# instruct Cmake to run Meta-Object Compiler (moc) automatically when needed
set(CMAKE_AUTOMOC ON)

# find the QtWidgets and OpenCV library
find_package(Qt5Widgets REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS} include 3rdparty)

file(GLOB SRC_FILES "src/*.cpp")
file(GLOB H_FILES "include/*.h")
add_library(ImageProcessing SHARED ${SRC_FILES} ${H_FILES})

# add the executable
add_executable(main main.cpp)

# link the executable with the different libraries
target_link_libraries(main ImageProcessing Qt5::Widgets ${OpenCV_LIBS} OpenCL)
