#ifndef __DESCRIPTOR_H__
#define __DESCRIPTOR_H__

#include <cv.h>
#include <vector>

class Descriptor {
    public:
        virtual cv::Mat getDescriptors(std::vector<cv::KeyPoint>& keypoints) = 0;
};

#endif

