#include "harris_detector.h"
#include <cv.h>
#include <highgui.h>
#include <iostream>

int HarrisDetector::BLOCK_SIZE = 3;
int HarrisDetector::APERTURE_SIZE = 3;
double HarrisDetector::K = 0.04;
int HarrisDetector::THRESHOLD = 127;

HarrisDetector* HarrisDetector::instance = NULL;

std::vector<cv::KeyPoint> HarrisDetector::run(cv::Mat& img) {
    std::vector<cv::KeyPoint> res;

    cv::GoodFeaturesToTrackDetector::Params params;
    params.useHarrisDetector = true;
    params.blockSize = HarrisDetector::BLOCK_SIZE;
    params.qualityLevel = HarrisDetector::K;
    params.minDistance = HarrisDetector::APERTURE_SIZE;

    cv::GoodFeaturesToTrackDetector detector(params);
    detector.detect(img, res);

    return res;
}

HarrisDetector* HarrisDetector::getInstance() {
    if(!HarrisDetector::instance) {
        HarrisDetector::instance = new HarrisDetector();
    }

    return HarrisDetector::instance;
}
