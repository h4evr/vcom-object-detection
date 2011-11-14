#ifndef __HARRIS_DETECTOR_H__
#define __HARRIS_DETECTOR_H__

#include "detector.h"

class HarrisDetector : public Detector {
    protected:
        static HarrisDetector* instance;

    public:
        /**
         * Detect features on an image.
         * @param img Image to extract feature from.
         * @returns Vector with feature points.
         */
        std::vector<cv::KeyPoint> run(cv::Mat& img);

        static HarrisDetector* getInstance();
};

#endif

