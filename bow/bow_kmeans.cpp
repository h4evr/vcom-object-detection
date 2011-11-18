#include "bow_kmeans.h"

BOWKMeans::BOWKMeans(int cluster_count) :
    trainer(cluster_count) {
}

void BOWKMeans::add(cv::Mat& descriptors) {
    trainer.add(descriptors);
}

cv::Mat BOWKMeans::run() {
    return trainer.cluster();
}

