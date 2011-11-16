FIND_PACKAGE(OpenCV REQUIRED)
ADD_LIBRARY(sift_descriptor sift_descriptor.cpp)
TARGET_LINK_LIBRARIES(sift_descriptor ${OpenCV_LIBS})
