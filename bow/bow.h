#ifndef __BOW_H__
#define __BOW_H__

#include <cv.h>

class BOW {
    public:
        virtual void add(cv::Mat& descriptors) = 0;
        virtual cv::Mat run() = 0;
};

#endif

