#include <cv.h>
#include <highgui.h>
#include <iostream>

#include "detection/harris_detector.h"
#include "detection/surf_detector.h"
#include "description/surf_descriptor.h"

#define USE_HARRIS 1

int main(int argc, char* argv[]) {
    cv::Mat img = cv::imread("caltech_data/airplanes_train/img001.jpg", -1);

    cv::Mat gray;

    cv::cvtColor(img, gray, CV_BGR2GRAY);

    //cv::namedWindow("original");
    //cv::imshow("original", img);

#if USE_HARRIS
    Detector* detector = HarrisDetector::getInstance();
    HarrisDetector::BLOCK_SIZE = 16;
    HarrisDetector::THRESHOLD = 100;
    HarrisDetector::K = 0.02;
#else
    Detector* detector = SURFDetector::getInstance();
#endif

    std::vector<cv::KeyPoint> key_points = detector->run(gray);

    cv::Mat descriptors = SURFDescriptor::getInstance()->getDescriptors(gray, key_points);

    std::cout << "Descriptors: " << descriptors.rows << std::endl;

    for(int i = 0; i < descriptors.rows; ++i) {
        for(int j = 0; j < descriptors.cols; ++j) {
            std::cout << descriptors.at<float>(i, j) << " ";
        }
        std::cout << std::endl;
    }

    cv::Mat output;
    cv::drawKeypoints(img, key_points, output);

    cv::namedWindow("harris");
    cv::imshow("harris", output);

    cv::waitKey();

    return 0;
}

