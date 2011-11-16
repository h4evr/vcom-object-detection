#include "surf_descriptor.h"

SURFDescriptor* SURFDescriptor::instance = NULL;

cv::Mat SURFDescriptor::getDescriptors(cv::Mat& img, std::vector<cv::KeyPoint>& keyPoints) {
    cv::Mat descriptors;
    cv::SurfDescriptorExtractor extractor;

    extractor.compute( img, keyPoints, descriptors);

    return descriptors;
}

SURFDescriptor* SURFDescriptor::getInstance() {
    if(!SURFDescriptor::instance) {
        SURFDescriptor::instance = new SURFDescriptor();
    }

    return instance;
}

