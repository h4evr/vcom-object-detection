#include "surf_descriptor.h"

SURFDescriptor* SURFDescriptor::instance = NULL;

SURFDescriptor::SURFDescriptor() :
    extractor(new cv::SurfDescriptorExtractor()) {
}

cv::Mat SURFDescriptor::getDescriptors(cv::Mat& img, std::vector<cv::KeyPoint>& keyPoints) {
    cv::Mat descriptors;

    extractor->compute( img, keyPoints, descriptors);

    return descriptors;
}

SURFDescriptor* SURFDescriptor::getInstance() {
    if(!SURFDescriptor::instance) {
        SURFDescriptor::instance = new SURFDescriptor();
    }

    return instance;
}

cv::DescriptorExtractor* SURFDescriptor::getOpenCVDescriptor() {
    return (cv::DescriptorExtractor*)extractor;
}

