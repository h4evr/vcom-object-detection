#ifndef __DETECTOR_H__
#define __DETECTOR_H__

#include <cv.h>
#include <vector>

/**
 * Abstract class for a feature detector.
 * All detectors should implement this.
 */
class Detector {
    public:
        /**
         * Detect features on an image.
         * @param img Image to extract feature from.
         * @returns Vector with feature points.
         */
        virtual std::vector<cv::KeyPoint> run(cv::Mat& img) = 0;
};

#endif

