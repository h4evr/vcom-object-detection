#include "flann_matcher.h"

FlannMatcher* FlannMatcher::instance = NULL;

FlannMatcher::FlannMatcher() : 
    matcher(new cv::FlannBasedMatcher()) {
}

FlannMatcher* FlannMatcher::getInstance() {
    if(!FlannMatcher::instance) {
        FlannMatcher::instance = new FlannMatcher();
    }

    return FlannMatcher::instance;
}

std::vector<cv::DMatch> FlannMatcher::match(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors) {
    std::vector<cv::DMatch> res;
    matcher->match(queryDescriptors, trainDescriptors, res);
    return res;
}

cv::DescriptorMatcher* FlannMatcher::getOpenCVMatcher() {
    return (cv::DescriptorMatcher*)matcher;
}

