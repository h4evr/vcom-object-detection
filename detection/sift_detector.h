#ifndef __SIFT_DETECTOR_H__
#define __SIFT_DETECTOR_H__

#include "detector.h"

class SIFTDetector : public Detector {
    protected:
        static SIFTDetector* instance;

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
        static SIFTDetector* getInstance();

        static int THRESHOLD;
        static int EDGE_THRESHOLD;
};



#endif

