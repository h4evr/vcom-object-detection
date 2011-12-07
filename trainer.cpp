#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <iostream>
#include <fstream>
#include <map>

#include "detection/harris_detector.h"
#include "detection/surf_detector.h"
#include "detection/sift_detector.h"

#include "description/surf_descriptor.h"
#include "description/sift_descriptor.h"

#include "matching/bruteforce_matcher.h"
#include "matching/flann_matcher.h"

#include "bow/bow_kmeans.h"
#include "bow/bow_descriptor.h"

#include "timer.h"

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
    std::cerr << "Usage: trainer confs_file list_of_images.txt num_of_cluster_centers out_vocabulary_file out_svm_path out_responses_path" << std::endl;
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
            exit(-1);
        }
        in.close();
    } else {
        std::cerr << "Couldn't open configuration file! Exiting!" << std::endl;
        exit(-1);
    }
}

int main(int argc, char* argv[]) {
    if(argc != 7) {
        usage();
        return -1;
    }

    Timer tmr;

    const char* confs_file = argv[1];
    const char* file = argv[2];
    int num_of_clusters = atoi(argv[3]);
    const char* out_voc = argv[4];
    const char* out_file = argv[5];
    const char* resp_file = argv[6];

    loadConfs(confs_file);

    std::map<std::string, std::vector<std::string> > class_images = load_list_of_files(file);

    cv::Mat gray;

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

    BOWKMeans trainer(num_of_clusters);

    std::vector<cv::KeyPoint> key_points;
    cv::Mat descriptors;

    std::map<std::string, std::vector<std::string> >::iterator it = class_images.begin();
    std::vector<std::string> images;

    unsigned int totalImages = 0;

    float descriptors_time = 0.0f;
    float clustering_time = 0.0f;
    float training_time = 0.0f;

    for(; it != class_images.end(); ++it) {

        std::cerr << "Analyzing images of class: " << it->first << std::endl;

        images = it->second;

        for(size_t i = 0; i < images.size(); ++i) {
            ++totalImages;

            std::cerr << (int)(((double)i / (double)images.size()) * 100.0) << "% Loading image: " << images[i] << std::endl;
            gray = cv::imread(images[i].c_str(), 0);

            tmr.Start();
            key_points = detector->run(gray);
            descriptors = descriptor->getDescriptors(gray, key_points);
            tmr.Stop();
            descriptors_time += tmr.GetElapsedTime();

            trainer.add(descriptors);

            gray.release();
            descriptors.release();
        }
    }

    if(totalImages > 0) {
        std::cerr << "Clustering centers..." << std::endl;
        tmr.Start();
        cv::Mat vocabulary = trainer.run();
        tmr.Stop();
        clustering_time = tmr.GetElapsedTime();

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

        //dumpDescriptors<float>(std::cout, trainData);

        std::cerr << "Training the SVM..." << std::endl;

        tmr.Start();
        svm.train(trainData, responses, cv::Mat(), cv::Mat(), params);
        tmr.Stop();
        training_time = tmr.GetElapsedTime();
        svm.save(out_file);

        strm_responses.close();
    }

    std::cerr << "TIME DESCRIPTION: " << descriptors_time << std::endl;
    std::cerr << "TIME CLUSTERING: " << clustering_time << std::endl;
    std::cerr << "TIME TRAINING: " << training_time << std::endl;

    delete matcher;
    delete detector;
    delete descriptor;

    return 0;
}

