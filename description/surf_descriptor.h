#ifndef __SURF_DESCRIPTOR_H__
#define __SURF_DESCRIPTOR_H__

#include "descriptor.h"

class SURFDescriptor : public Descriptor {
    protected:
        static SURFDescriptor* instance;
        cv::Ptr<cv::SurfDescriptorExtractor> extractor;

    public:
        SURFDescriptor();

        cv::Mat getDescriptors(cv::Mat& img, std::vector<cv::KeyPoint>& keyPoints);
        cv::DescriptorExtractor* getOpenCVDescriptor();
        static SURFDescriptor* getInstance();
};

#endif

