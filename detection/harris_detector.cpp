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

    cv::Mat corners = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::Mat corners_norm, corners_norm_scaled;

    cv::cornerHarris(img,
                     corners,
                     HarrisDetector::BLOCK_SIZE,
                     HarrisDetector::APERTURE_SIZE,
                     HarrisDetector::K,
                     cv::BORDER_DEFAULT);

    cv::normalize(corners, corners_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(corners_norm, corners_norm_scaled);

    for(int j = 0; j < corners_norm_scaled.rows; ++j) {
        for(int i = 0; i < corners_norm_scaled.cols; ++i) {
            if((int)corners_norm.at<float>(j, i) >= HarrisDetector::THRESHOLD) {
                res.push_back(cv::KeyPoint(cv::Point2f((float)i, (float)j), 0, -1, 0, -1));
            }
        }
    }

    return res;
}

HarrisDetector* HarrisDetector::getInstance() {
    if(!HarrisDetector::instance) {
        HarrisDetector::instance = new HarrisDetector();
    }

    return HarrisDetector::instance;
}
