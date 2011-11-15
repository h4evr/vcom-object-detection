FIND_PACKAGE(OpenCV REQUIRED)
ADD_LIBRARY(harris_detector harris_detector.cpp)
TARGET_LINK_LIBRARIES(harris_detector ${OpenCV_LIBS})
