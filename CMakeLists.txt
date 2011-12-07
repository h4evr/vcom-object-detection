cmake_minimum_required(VERSION 2.8)
PROJECT(vcom-object-detection)
FIND_PACKAGE(OpenCV REQUIRED)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(SRCS main.cpp)

SET(LIBS harris_detector surf_detector sift_detector surf_descriptor sift_descriptor bruteforce_matcher flann_matcher bow_descriptor timer)

#SET(CMAKE_CXX_FLAGS "-Wall -g -pg")

ADD_SUBDIRECTORY(detection)
ADD_SUBDIRECTORY(description)
ADD_SUBDIRECTORY(matching)
ADD_SUBDIRECTORY(bow)

ADD_LIBRARY(timer timer.cpp)
ADD_EXECUTABLE(vcom-object-detection ${SRCS})
ADD_EXECUTABLE(trainer trainer.cpp)
TARGET_LINK_LIBRARIES(vcom-object-detection ${OpenCV_LIBS} ${LIBS})
TARGET_LINK_LIBRARIES(trainer ${OpenCV_LIBS} ${LIBS} bow_kmeans)
