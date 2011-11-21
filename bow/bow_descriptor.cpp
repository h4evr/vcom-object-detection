#include "bow_descriptor.h"

BOWDescriptor::BOWDescriptor(Descriptor* descriptor, Matcher* matcher, cv::Mat& vocabulary) :
    extractor(new cv::BOWImgDescriptorExtractor(descriptor->getOpenCVDescriptor(), matcher->getOpenCVMatcher()))
{
    extractor->setVocabulary(vocabulary);
}

cv::Mat BOWDescriptor::extractBOWHistogram(cv::Mat& img, std::vector<cv::KeyPoint>& keyPoints) {
    cv::Mat out;
    extractor->compute(img, keyPoints, out);
    return out;
}

