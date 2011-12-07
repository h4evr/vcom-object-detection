#ifndef __SIFT_DESCRIPTOR_H__
#define __SIFT_DESCRIPTOR_H__

#include "descriptor.h"

class SIFTDescriptor : public Descriptor {
    protected:
        static SIFTDescriptor* instance;
        cv::Ptr<cv::SiftDescriptorExtractor> extractor;

    public:
        SIFTDescriptor();

        cv::Mat getDescriptors(cv::Mat& img, std::vector<cv::KeyPoint>& keyPoints);
        cv::DescriptorExtractor* getOpenCVDescriptor();
        static SIFTDescriptor* getInstance();
};

#endif

