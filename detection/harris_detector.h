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

        /**
         * Retrieve the singleton instance of this Detector.
         * @returns Instance of Detector.
         */
        static HarrisDetector* getInstance();

        /**
         * The neighborhood size parameter to pass to the harris corner algorithm.
         */
        static int BLOCK_SIZE;

        /**
         * The aperture size for the Sobel operator.
         */
        static int APERTURE_SIZE;

        /**
         * The free parameter to pass to the harris corner algorithm.
         */
        static double K;

        /**
         * The threshold value to filter out bad corners.
         */
        static int THRESHOLD;

};

#endif

