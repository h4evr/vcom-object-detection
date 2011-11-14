#include "harris_detector.h"

HarrisDetector* HarrisDetector::instance = NULL;

std::vector<cv::KeyPoint> HarrisDetector::run(cv::Mat& img) {
    std::vector<cv::KeyPoint> res;

    return res;
}

HarrisDetector* HarrisDetector::getInstance() {
    if(!HarrisDetector::instance) {
        HarrisDetector::instance = new HarrisDetector();
    }
    
    return HarrisDetector::instance;
}
