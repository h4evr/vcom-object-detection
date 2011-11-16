cmake_minimum_required(VERSION 2.8)
PROJECT(vcom-object-detection)
FIND_PACKAGE(OpenCV REQUIRED)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(SRCS main.cpp)

ADD_SUBDIRECTORY(detection)
ADD_SUBDIRECTORY(description)
ADD_SUBDIRECTORY(matching)

ADD_EXECUTABLE(vcom-object-detection ${SRCS})
TARGET_LINK_LIBRARIES(vcom-object-detection ${OpenCV_LIBS} harris_detector surf_detector surf_descriptor bruteforce_matcher)
