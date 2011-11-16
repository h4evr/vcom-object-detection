#include "surf_detector.h"
#include <cv.h>
#include <highgui.h>
#include <iostream>

int SURFDetector::MIN_HESSIAN = 400;

SURFDetector* SURFDetector::instance = NULL;

std::vector<cv::KeyPoint> SURFDetector::run(cv::Mat& img) {
    std::vector<cv::KeyPoint> res;

    cv::SurfFeatureDetector detector(SURFDetector::MIN_HESSIAN);
    detector.detect(img, res);

    return res;
}

SURFDetector* SURFDetector::getInstance() {
    if(!SURFDetector::instance) {
        SURFDetector::instance = new SURFDetector();
    }

    return SURFDetector::instance;
}

