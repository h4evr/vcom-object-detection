#include "flann_matcher.h"

FlannMatcher* FlannMatcher::instance = NULL;

FlannMatcher* FlannMatcher::getInstance() {
    if(!FlannMatcher::instance) {
        FlannMatcher::instance = new FlannMatcher();
    }

    return FlannMatcher::instance;
}

std::vector<cv::DMatch> FlannMatcher::match(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors) {
    std::vector<cv::DMatch> res;
    cv::FlannBasedMatcher matcher;
    matcher.match(queryDescriptors, trainDescriptors, res);
    return res;
}

