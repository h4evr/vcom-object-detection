#ifndef __BOW_KMEANS_H__
#define __BOW_KMEANS_H__

#include "bow.h"
#include <cv.h>

class BOWKMeans : public BOW {
    public:
        BOWKMeans(int cluster_count);

        void add(cv::Mat& descriptors);
        cv::Mat run();
    private:
        cv::BOWKMeansTrainer trainer;
};

#endif

