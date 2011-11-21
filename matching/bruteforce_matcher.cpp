#include "bruteforce_matcher.h"

BruteForceMatcher* BruteForceMatcher::instance = NULL;

BruteForceMatcher::BruteForceMatcher() : 
    matcher(new cv::BruteForceMatcher<cv::L2<float> >())
{
}

BruteForceMatcher* BruteForceMatcher::getInstance() {
    if(!BruteForceMatcher::instance) {
        BruteForceMatcher::instance = new BruteForceMatcher();
    }

    return BruteForceMatcher::instance;
}

std::vector<cv::DMatch> BruteForceMatcher::match(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors) {
    std::vector<cv::DMatch> res;
    matcher->match(queryDescriptors, trainDescriptors, res);
    return res;
}

cv::DescriptorMatcher* BruteForceMatcher::getOpenCVMatcher() {
    return (cv::DescriptorMatcher*) matcher;
}

