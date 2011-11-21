#ifndef __FLANN_MATCHER_H__
#define __FLANN_MATCHER_H__

#include "matcher.h"

class FlannMatcher : public Matcher {
    protected:
        static FlannMatcher* instance;
        cv::Ptr<cv::FlannBasedMatcher> matcher;

    public:
        FlannMatcher();
        std::vector<cv::DMatch> match(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors);
        static FlannMatcher* getInstance();
        
        cv::DescriptorMatcher* getOpenCVMatcher();
};

#endif

