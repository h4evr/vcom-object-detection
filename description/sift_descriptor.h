#ifndef __SIFT_DESCRIPTOR_H__
#define __SIFT_DESCRIPTOR_H__

#include "descriptor.h"

class SIFTDescriptor : public Descriptor {
    protected:
        static SIFTDescriptor* instance;

    public:
        cv::Mat getDescriptors(std::vector<cv::KeyPoint>& keyPoints);
        static SIFTDescriptor* getInstance();
};

#endif

