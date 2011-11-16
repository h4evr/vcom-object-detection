FIND_PACKAGE(OpenCV REQUIRED)
ADD_LIBRARY(surf_descriptor surf_descriptor.cpp)
TARGET_LINK_LIBRARIES(surf_descriptor ${OpenCV_LIBS})
