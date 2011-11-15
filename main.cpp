#include <cv.h>
#include <highgui.h>

#include "detection/harris_detector.h"

void drawKeyPoints(cv::Mat& img, std::vector<cv::KeyPoint>& keypoints) {
    for(size_t i = 0; i < keypoints.size(); ++i) {
        cv::Point2f point = keypoints[i].pt;
        cv::circle(img, cv::Point((int)point.x, (int)point.y), 5, cv::Scalar(0, 0, 255), 1, 8, 0);
    }
}

int main(int argc, char* argv[]) {
    cv::Mat img = cv::imread("caltech_data/airplanes_train/img001.jpg", -1);

    cv::namedWindow("original");
    cv::imshow("original", img);

    Detector* detector = HarrisDetector::getInstance();

    HarrisDetector::BLOCK_SIZE = 8;
    HarrisDetector::THRESHOLD = 150;
    HarrisDetector::K = 0.04;

    std::vector<cv::KeyPoint> key_points = detector->run(img);

    cv::Mat output = img.clone();
    drawKeyPoints(output, key_points);

    cv::namedWindow("harris");
    cv::imshow("harris", output);

    cv::waitKey();

    return 0;
}

