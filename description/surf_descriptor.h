#ifndef __SURF_DESCRIPTOR_H__
#define __SURF_DESCRIPTOR_H__

#include "descriptor.h"

class SURFDescriptor : public Descriptor {
    protected:
        static SURFDescriptor* instance;

    public:
        cv::Mat getDescriptors(cv::Mat& img, std::vector<cv::KeyPoint>& keyPoints);
        static SURFDescriptor* getInstance();
};

#endif

