#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <iostream>
#include <fstream>
#include <map>

#include "detection/harris_detector.h"
#include "detection/surf_detector.h"

#include "description/surf_descriptor.h"

#include "matching/bruteforce_matcher.h"
#include "matching/flann_matcher.h"

#include "bow/bow_kmeans.h"
#include "bow/bow_descriptor.h"

#define USE_HARRIS_DETECTOR 0
#define USE_BRUTEFORCE_MATCHER 0


std::map<std::string, std::vector<std::string> > load_list_of_files(const char* file) {
    std::vector<std::string> files;
    std::map<std::string, std::vector<std::string> > res;
    std::string buffer;
    std::string className;

    std::ifstream in(file);

    while(!in.eof()) {
        std::getline(in, buffer);
        if(!buffer.empty()) {
            if(buffer[0] == '-') {
                if(files.size() > 0) {
                    if(className.empty()) {
                        className = "class 1";
                    }

                    res.insert(std::pair<std::string, std::vector<std::string> >(className, files));

                    files.clear();
                }

                className = buffer.substr(1);
            } else {
                files.push_back(buffer);
            }
        }
    }

    if(files.size() > 0) {
        if(className.empty()) {
            className = "class 1";
        }

        res.insert(std::pair<std::string, std::vector<std::string> >(className, files));
    }

    in.close();

    return res;
}

template<class T>
void dumpDescriptors(std::ostream& strm, cv::Mat& descriptors) {
    strm << descriptors.cols << " " << descriptors.rows << std::endl;
    for(int i = 0; i < descriptors.rows; ++i) {
        for(int j = 0; j < descriptors.cols; ++j) {
            strm << descriptors.at<T>(i, j) << " ";
        }
        strm << std::endl;
    }
}

void usage() {
    std::cerr << "Usage: trainer list_of_images.txt num_of_cluster_centers out_vocabulary_file out_svm_path out_responses_path" << std::endl;
}

cv::Mat collectTrainData(Detector* detector, cv::Ptr<cv::BOWImgDescriptorExtractor> extractor, std::vector<std::string>& images) {
    std::vector<cv::Mat> descriptors;
    descriptors.reserve(images.size());

    for(size_t i = 0; i < images.size(); ++i) {
        cv::Mat image_gray = cv::imread(images[i], 0);
        std::vector<cv::KeyPoint> keyPoints = detector->run(image_gray);
        cv::Mat descriptor;

        extractor->compute(image_gray, keyPoints, descriptor);

        if(descriptor.empty()) {
            std::cerr << "Removing image, no descriptors: " << images[i] << std::endl;
            images.erase(images.begin() + i);
            --i;
        } else {
            descriptors.push_back(descriptor);
        }
    }

    int numCols = extractor->getVocabulary().rows;
    cv::Mat trainData(descriptors.size(), numCols, CV_32FC1);

    for(size_t i = 0; i < descriptors.size(); ++i) {
        for(size_t j = 0; j < numCols; ++j) {
            trainData.at<float>(i, j) = descriptors[i].at<float>(0, j);
        }
    }

    return trainData;
}

int main(int argc, char* argv[]) {
    if(argc != 6) {
        usage();
        return -1;
    }

    const char* file = argv[1];
    int num_of_clusters = atoi(argv[2]);
    const char* out_voc = argv[3];
    const char* out_file = argv[4];
    const char* resp_file = argv[5];
    std::map<std::string, std::vector<std::string> > class_images = load_list_of_files(file);

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
    Matcher* matcher = FlannMatcher::getInstance();

    BOWKMeans trainer(num_of_clusters);

    std::vector<cv::KeyPoint> key_points;
    cv::Mat descriptors;

    std::map<std::string, std::vector<std::string> >::iterator it = class_images.begin();
    std::vector<std::string> images;

    unsigned int totalImages = 0;

    for(; it != class_images.end(); ++it) {

        std::cerr << "Analyzing images of class: " << it->first << std::endl;

        images = it->second;

        for(size_t i = 0; i < images.size(); ++i) {
            ++totalImages;

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
    }

    if(totalImages > 0) {
        std::cerr << "Clustering centers..." << std::endl;
        cv::Mat vocabulary = trainer.run();

        std::ofstream of(out_voc);
        if(of.is_open()) {

            dumpDescriptors<float>(of, vocabulary);
            of.close();
        }

        cv::Ptr<cv::BOWImgDescriptorExtractor> extractor;

        extractor = new cv::BOWImgDescriptorExtractor(descriptor->getOpenCVDescriptor(), matcher->getOpenCVMatcher());
        extractor->setVocabulary(vocabulary);

        cv::SVM svm;
        cv::SVMParams params;
        params.svm_type = cv::SVM::C_SVC;
        params.kernel_type = cv::SVM::RBF;
        params.C = 1;
        params.term_crit = cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1000, 0.00001);

        int classId = 0, row = 0, totalRows = 0;

        cv::Mat trainData(totalImages, vocabulary.rows, CV_32FC1);
        cv::Mat responses(totalImages, 1, CV_32SC1);

        std::ofstream strm_responses(resp_file);

        it = class_images.begin();
        for(; it != class_images.end(); ++it) {
            ++classId;
            strm_responses << classId << " " << it->first << std::endl;

            std::cerr << "Calculating vocabulary descriptors for class: " << it->first << std::endl;
            std::vector<std::string> images = it->second;

            cv::Mat classTrainData = collectTrainData(detector, extractor, images);
            cv::Mat classResponses(classTrainData.rows, 1, CV_32SC1);

            for(int i = 0; i < classTrainData.rows; ++i) {
                for(int j = 0; j < classTrainData.cols; ++j) {
                    trainData.at<float>(row + i, j) = classTrainData.at<float>(i, j);
                }
                responses.at<int>(row + i, 0) = classId;
            }

            row += classTrainData.rows;
            totalRows += classTrainData.rows;
        }

        trainData.resize(totalRows);
        responses.resize(totalRows);

        dumpDescriptors<float>(std::cout, trainData);

        std::cerr << "Training the SVM..." << std::endl;
        svm.train(trainData, responses, cv::Mat(), cv::Mat(), params);
        svm.save(out_file);

        strm_responses.close();
    }

    delete matcher;
    delete detector;
    delete descriptor;

    return 0;
}

