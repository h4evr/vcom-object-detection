#ifndef __SURF_DETECTOR_H__
#define __SURF_DETECTOR_H__

#include "detector.h"

class SURFDetector : public Detector {
    protected:
        static SURFDetector* instance;

    public:
        /**
         * Detect features on an image.
         * @param img Image to extract feature from.
         * @returns Vector with feature points.
         */
        std::vector<cv::KeyPoint> run(cv::Mat& img);

        /**
         * Retrieve the singleton instance of this Detector.
         * @returns Instance of Detector.
         */
        static SURFDetector* getInstance();

        static int MIN_HESSIAN;
};



#endif

