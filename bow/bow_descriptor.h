#ifndef __BOW_DESCRIPTOR_H__
#define __BOW_DESCRIPTOR_H__

#include <cv.h>
#include <vector>

#include "../description/descriptor.h"
#include "../matching/matcher.h"

class BOWDescriptor {
    public:
        BOWDescriptor(Descriptor* descriptor, Matcher* matcher, cv::Mat& vocabulary);
        cv::Mat extractBOWHistogram(cv::Mat& img, std::vector<cv::KeyPoint>& keyPoints);
    private:
        cv::Ptr<cv::BOWImgDescriptorExtractor> extractor;
        cv::Mat vocabulary;
};

#endif
