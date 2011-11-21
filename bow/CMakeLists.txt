FIND_PACKAGE(OpenCV REQUIRED)

ADD_LIBRARY(bow_kmeans bow_kmeans.cpp)
ADD_LIBRARY(bow_descriptor bow_descriptor.cpp)

TARGET_LINK_LIBRARIES(bow_kmeans ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(bow_descriptor ${OpenCV_LIBS})
