#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>

#include "detection/harris_detector.h"
#include "detection/surf_detector.h"
#include "detection/sift_detector.h"

#include "description/surf_descriptor.h"
#include "description/sift_descriptor.h"

#include "matching/bruteforce_matcher.h"
#include "matching/flann_matcher.h"

#include "bow/bow_descriptor.h"

// CONF INDEX:
//   0: detector
//   1: descriptor
//   2: matcher
// CONF DETECTOR:
//   0: HARRIS
//   1: SURF
//   2: SIFT
// CONF DESCRIPTOR:
//   0: SIFT
//   1: SURF
// CONF MATCHER:
//   0: BRUTEFORCE
//   1: FLANN BASED
int confs[] = { 1, 1, 1 };

template<class T>
void dumpDescriptors(cv::Mat& descriptors) {
    std::cout << "Descriptors: " << descriptors.rows << std::endl;

    for(int i = 0; i < descriptors.rows; ++i) {
        for(int j = 0; j < descriptors.cols; ++j) {
            std::cout << descriptors.at<T>(i, j) << " ";
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

cv::Mat loadInputData(const char* file, Detector* detector, cv::Ptr<cv::BOWImgDescriptorExtractor> extractor) {
    //std::cerr << "Loading file " << file << std::endl;
    cv::Mat img = cv::imread(file, 0);
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptor;

    keyPoints = detector->run(img);
    extractor->compute(img, keyPoints, descriptor);

    return descriptor;
}

std::map<int, std::string> loadResponses(const char* filename) {
    std::map<int, std::string> res;
    std::ifstream in(filename);

    if(in.is_open()) {
        int resp;
        std::string className;

        while(!in.eof()) {
            in >> resp;
            in >> className;
            res.insert(std::pair<int, std::string>(resp, className));
        }

        in.close();
    }

    return res;
}

void loadConfs(const char* filename) {
    std::ifstream in(filename);
    std::string buffer;

    if(in.is_open()) {
        if(!in.eof()) {
            in >> buffer;
            if(buffer.compare("harris") == 0) {
                confs[0] = 0;
            } else if(buffer.compare("surf") == 0) {
                confs[0] = 1;
            } else if(buffer.compare("sift") == 0) {
                confs[0] = 2;
            }
        } else {
            std::cerr << "Invalid configuration file!" << std::endl;
            in.close();
            exit(-1);
        }

        if(!in.eof()) {
            in >> buffer;
            if(buffer.compare("sift") == 0) {
                confs[1] = 0;
            } else if(buffer.compare("surf") == 0) {
                confs[1] = 1;
            }
        } else {
            std::cerr << "Invalid configuration file!" << std::endl;
            in.close();
            exit(-1);
        }

        if(!in.eof()) {
            in >> buffer;
            if(buffer.compare("bruteforce") == 0) {
                confs[2] = 0;
            } else if(buffer.compare("flann") == 0) {
                confs[2] = 1;
            }
        } else {
            std::cerr << "Invalid configuration file!" << std::endl;
            in.close();
            exit(-1);
        }

        in.close();
    } else {
        std::cerr << "Couldn't open configuration file! Exiting!" << std::endl;
        exit(-1);
    }
}

int main(int argc, char* argv[]) {

    if(argc <= 5) {
        std::cerr << "Usage: ./vcom-object-detection conf_file vocabulary svm responses file1 [file2] [...]" << std::endl;
        return -1;
    }

    const char* conf_file = argv[1];
    const char* voc_file = argv[2];
    const char* svm_file = argv[3];
    const char* res_file = argv[4];

    loadConfs(conf_file);

    std::map<int, std::string> resp_to_class = loadResponses(res_file);

    Detector* detector;

    switch(confs[0]) {
    case 0:
        detector = HarrisDetector::getInstance();
        HarrisDetector::BLOCK_SIZE = 16;
        HarrisDetector::THRESHOLD = 100;
        HarrisDetector::K = 0.02;
        break;
    case 1:
        detector = SURFDetector::getInstance();
        break;

    case 2:
        detector = SIFTDetector::getInstance();
        break;
    };

    Descriptor* descriptor;

    switch(confs[1]) {
    case 0:
        descriptor = SIFTDescriptor::getInstance();
        break;
    case 1:
        descriptor = SURFDescriptor::getInstance();
        break;
    };

    Matcher* matcher;

    switch(confs[2]) {
        case 0:
            matcher = BruteForceMatcher::getInstance();
            break;
        case 1:
            matcher = FlannMatcher::getInstance();
            break;
    };

    cv::Mat vocabulary = loadVocabulary(voc_file);

    cv::Ptr<cv::BOWImgDescriptorExtractor> extractor = new cv::BOWImgDescriptorExtractor(descriptor->getOpenCVDescriptor(), matcher->getOpenCVMatcher());
    extractor->setVocabulary(vocabulary);

    cv::SVM svm;
    svm.load(svm_file);

    cv::Mat inputData;

    for(int i = 5; i < argc; ++i) {
        inputData = loadInputData(argv[i], detector, extractor);
        int resp = (int)svm.predict(inputData, false);
        std::cout << argv[i] << ": " << resp_to_class[resp] << " (" << resp << ")" << std::endl;
    }

    delete matcher;
    delete descriptor;
    delete detector;

    return 0;
}

