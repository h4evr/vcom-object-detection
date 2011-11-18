FIND_PACKAGE(OpenCV REQUIRED)
ADD_LIBRARY(bow_kmeans bow_kmeans.cpp)
TARGET_LINK_LIBRARIES(bow_kmeans ${OpenCV_LIBS})
