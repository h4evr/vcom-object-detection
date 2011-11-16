#include "sift_descriptor.h"

SIFTDescriptor* SIFTDescriptor::instance = NULL;

cv::Mat SIFTDescriptor::getDescriptors(std::vector<cv::KeyPoint>& keyPoints) {
    cv::Mat res;

    return res;
}

SIFTDescriptor* SIFTDescriptor::getInstance() {
    if(!SIFTDescriptor::instance) {
        SIFTDescriptor::instance = new SIFTDescriptor();
    }

    return instance;
}

