#include "sift_descriptor.h"

SIFTDescriptor* SIFTDescriptor::instance = NULL;

SIFTDescriptor::SIFTDescriptor() :
    extractor(new cv::SiftDescriptorExtractor()) {
}

cv::Mat SIFTDescriptor::getDescriptors(cv::Mat& img, std::vector<cv::KeyPoint>& keyPoints) {
    cv::Mat descriptors;

    extractor->compute( img, keyPoints, descriptors);

    return descriptors;
}

SIFTDescriptor* SIFTDescriptor::getInstance() {
    if(!SIFTDescriptor::instance) {
        SIFTDescriptor::instance = new SIFTDescriptor();
    }

    return instance;
}

cv::DescriptorExtractor* SIFTDescriptor::getOpenCVDescriptor() {
    return (cv::DescriptorExtractor*)extractor;
}

