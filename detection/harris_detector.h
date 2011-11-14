#ifndef __HARRIS_DETECTOR_H__
#define __HARRIS_DETECTOR_H__

#include "detector.h"

class HarrisDetector : public Detector {
    public:
        /**
         * Detect features on an image.
         * @param img Image to extract feature from.
         * @returns Vector with feature points.
         */
        std::vector<cv::Point2f> run(cv::Mat& img);
};

#endif

