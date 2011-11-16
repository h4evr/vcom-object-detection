FIND_PACKAGE(OpenCV REQUIRED)
ADD_LIBRARY(bruteforce_matcher bruteforce_matcher.cpp)
TARGET_LINK_LIBRARIES(bruteforce_matcher ${OpenCV_LIBS})
