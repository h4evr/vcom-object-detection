#ifndef __MATCHER_H__
#define __MATCHER_H__

#include <cv.h>
#include <vector>

class Matcher {

    public:
        virtual std::vector<cv::DMatch> match(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors) = 0;
};

#endif
