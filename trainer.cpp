#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <fstream>

#include "detection/harris_detector.h"
#include "detection/surf_detector.h"

#include "description/surf_descriptor.h"

#include "matching/bruteforce_matcher.h"
#include "matching/flann_matcher.h"

#include "bow/bow_kmeans.h"
#include "bow/bow_descriptor.h"

#define USE_HARRIS_DETECTOR 0
#define USE_BRUTEFORCE_MATCHER 0

std::vector<std::string> load_list_of_files(const char* file) {
    std::vector<std::string> files;
    std::string buffer;

    std::ifstream in(file);

    while(!in.eof()) {
        std::getline(in, buffer);
        if(!buffer.empty()) {
            files.push_back(buffer);
        }
    }

    in.close();

    return files;
}

void dumpDescriptors(cv::Mat& descriptors) {
    std::cout << descriptors.cols << " " << descriptors.rows << std::endl;
    for(int i = 0; i < descriptors.rows; ++i) {
        for(int j = 0; j < descriptors.cols; ++j) {
            std::cout << descriptors.at<float>(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

void usage() {
    std::cerr << "Usage: trainer list_of_images.txt num_of_cluster_centers > outfile" << std::endl;
}

int main(int argc, char* argv[]) {
    if(argc != 3) {
        usage();
        return -1;
    }

    const char* file = argv[1];
    int num_of_clusters = atoi(argv[2]);

    std::vector<std::string> images = load_list_of_files(file);

    cv::Mat img, gray;

#if USE_HARRIS_DETECTOR
    Detector* detector = HarrisDetector::getInstance();
    HarrisDetector::BLOCK_SIZE = 16;
    HarrisDetector::THRESHOLD = 100;
    HarrisDetector::K = 0.02;
#else
    Detector* detector = SURFDetector::getInstance();
#endif

    Descriptor* descriptor = SURFDescriptor::getInstance();

    BOWKMeans trainer(num_of_clusters);

    std::vector<cv::KeyPoint> key_points;
    cv::Mat descriptors;

    for(size_t i = 0; i < images.size(); ++i) {
        std::cerr << (int)(((double)i / (double)images.size()) * 100.0) << "% Loading image: " << images[i] << std::endl;
        img = cv::imread(images[i].c_str(), -1);
        cvtColor(img, gray, CV_BGR2GRAY);

        key_points = detector->run(gray);
        descriptors = descriptor->getDescriptors(gray, key_points);

        trainer.add(descriptors);

        gray.release();
        img.release();
        descriptors.release();
    }

    if(images.size() > 0) {
        std::cerr << "Clustering centers..." << std::endl;
        cv::Mat vocabulary = trainer.run();
        dumpDescriptors(vocabulary);
    }

    delete detector;
    delete descriptor;

    return 0;
}

