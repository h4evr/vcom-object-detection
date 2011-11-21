#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <fstream>
#include "detection/harris_detector.h"
#include "detection/surf_detector.h"

#include "description/surf_descriptor.h"

#include "matching/bruteforce_matcher.h"
#include "matching/flann_matcher.h"

#include "bow/bow_descriptor.h"

#define USE_HARRIS_DETECTOR 0
#define USE_BRUTEFORCE_MATCHER 0

void dumpDescriptors(cv::Mat& descriptors) {
    std::cout << "Descriptors: " << descriptors.rows << std::endl;

    for(int i = 0; i < descriptors.rows; ++i) {
        for(int j = 0; j < descriptors.cols; ++j) {
            std::cout << descriptors.at<float>(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

cv::Mat loadVocabulary(const char* filename) {

    std::ifstream in(filename);
    std::string buffer;

    int cols, rows;

    in >> cols;
    in >> rows;

    cv::Mat out(rows, cols, CV_32FC1);

    int j = 0;
    while(!in.eof()) {
        for(int i = 0; i < cols; ++i) {
            in >> out.at<float>(j, i);
        }
        ++j;
    }

    in.close();

    return out;
}

int main(int argc, char* argv[]) {
    cv::Mat img1 = cv::imread("caltech_data/airplanes_train/img001.jpg", -1);
    cv::Mat img2 = cv::imread("caltech_data/airplanes_train/img002.jpg", -1);

    cv::Mat gray1,
            gray2;

    cv::cvtColor(img1, gray1, CV_BGR2GRAY);
    cv::cvtColor(img2, gray2, CV_BGR2GRAY);

#if USE_HARRIS_DETECTOR
    Detector* detector = HarrisDetector::getInstance();
    HarrisDetector::BLOCK_SIZE = 16;
    HarrisDetector::THRESHOLD = 100;
    HarrisDetector::K = 0.02;
#else
    Detector* detector = SURFDetector::getInstance();
#endif

    Descriptor* descriptor = SURFDescriptor::getInstance();

#if USE_BRUTEFORCE_MATCHER
    Matcher* matcher = BruteForceMatcher::getInstance();
#else
    Matcher* matcher = FlannMatcher::getInstance();
#endif

    std::vector<cv::KeyPoint> key_points_1 = detector->run(gray1);
    cv::Mat descriptors_1 = descriptor->getDescriptors(gray1, key_points_1);

    std::vector<cv::KeyPoint> key_points_2 = detector->run(gray2);
    cv::Mat descriptors_2 = descriptor->getDescriptors(gray1, key_points_2);

    std::vector<cv::DMatch> matches = matcher->match(descriptors_1, descriptors_2);

    //dumpDescriptors(descriptors_1);
    //dumpDescriptors(descriptors_2);

    cv::Mat vocabulary = loadVocabulary("vocabulary.dat");

    BOWDescriptor bowdesc(descriptor, matcher, vocabulary);
    cv::Mat histDescriptor = bowdesc.extractBOWHistogram(gray1, key_points_1);

    dumpDescriptors(histDescriptor);

    cv::Mat output_1,
            output_2,
            output_matches;

    cv::drawKeypoints(img1, key_points_1, output_1);
    cv::drawKeypoints(img2, key_points_2, output_2);

    cv::drawMatches(img1, key_points_1,
                    img2, key_points_2,
                    matches,
                    output_matches);

    cv::namedWindow("output_1");
    cv::imshow("output_1", output_1);

    cv::namedWindow("output_2");
    cv::imshow("output_2", output_2);

    cv::namedWindow("output_matches");
    cv::imshow("output_matches", output_matches);

    cv::waitKey();

    delete detector;
    delete descriptor;
    delete matcher;

    return 0;
}

