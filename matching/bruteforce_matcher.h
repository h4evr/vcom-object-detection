#ifndef __BRUTEFORCE_MATCHER_H__
#define __BRUTEFORCE_MATCHER_H__

#include "matcher.h"

class BruteForceMatcher : public Matcher {
    protected:
        static BruteForceMatcher* instance;
        cv::Ptr<cv::DescriptorMatcher> matcher;

    public:
        BruteForceMatcher();
        std::vector<cv::DMatch> match(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors);
        static BruteForceMatcher* getInstance();
        cv::DescriptorMatcher* getOpenCVMatcher();
};

#endif

