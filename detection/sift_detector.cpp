#include "sift_detector.h"
#include <cv.h>
#include <highgui.h>
#include <iostream>

int SIFTDetector::THRESHOLD = 127;
int SIFTDetector::EDGE_THRESHOLD = 127;

SIFTDetector* SIFTDetector::instance = NULL;

std::vector<cv::KeyPoint> SIFTDetector::run(cv::Mat& img) {
    std::vector<cv::KeyPoint> res;

    cv::SiftFeatureDetector detector(SIFTDetector::THRESHOLD, SIFTDetector::EDGE_THRESHOLD);
    detector.detect(img, res);

    return res;
}

SIFTDetector* SIFTDetector::getInstance() {
    if(!SIFTDetector::instance) {
        SIFTDetector::instance = new SIFTDetector();
    }

    return SIFTDetector::instance;
}

