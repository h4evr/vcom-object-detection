#include <cv.h>
#include "detection/harris_detector.h"

int main(int argc, char* argv[]) {
    //std::cout << "Hello World!" << std::endl;

    Detector* detector = HarrisDetector::getInstance();

    cv::Mat img;

    std::vector<cv::KeyPoint> key_points = detector->run(img);

    return 0;
}

