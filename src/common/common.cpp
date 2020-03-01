#include "common.h"
#include <algorithm>
#include <iostream>
#include "opencv2/imgproc.hpp"


namespace mirror {

uint8_t* GetImage(const cv::Mat& img_src) {
    uchar* data_ptr = new uchar[img_src.total() * 4];
    cv::Mat img_tmp(img_src.size(), CV_8UC4, data_ptr);
    cv::cvtColor(img_src, img_tmp, cv::COLOR_BGR2RGBA, 4);
    return (uint8_t*)img_tmp.data;
}

float InterRectArea(const cv::Rect& a, const cv::Rect& b) {
    cv::Point left_top = cv::Point(MAX(a.x, b.x), MAX(a.y, b.y));
    cv::Point right_bottom = cv::Point(MIN(a.br().x, b.br().x), MIN(a.br().y, b.br().y));
    cv::Point diff = right_bottom - left_top;
    return (MAX(diff.x + 1, 0) * MAX(diff.y + 1, 0));
}

int ComputeIOU(const cv::Rect& rect1,
    const cv::Rect& rect2, float* iou,
    const std::string& type) {

    float inter_area = InterRectArea(rect1, rect2);
    if (type == "UNION") {
        *iou = inter_area / (rect1.area() + rect2.area() - inter_area);
    }
    else {
        *iou = inter_area / MIN(rect1.area(), rect2.area());
    }

    return 0;
}

float CalculateSimilarity(const std::vector<float>&feat1, const std::vector<float>& feat2) {
    if (feat1.size() != feat2.size()) {
		std::cout << "feature size not match." << std::endl;
		return 10003;
	}
	float inner_product = 0.0f;
	float feat_norm1 = 0.0f;
	float feat_norm2 = 0.0f;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(threads_num)
#endif
	for(int i = 0; i < kFaceFeatureDim; ++i) {
		inner_product += feat1[i] * feat2[i];
		feat_norm1 += feat1[i] * feat1[i];
		feat_norm2 += feat2[i] * feat2[i];
	}
	return inner_product / sqrt(feat_norm1) / sqrt(feat_norm2);
}

float Logists(const float& value) {
    return 1.0f / (1.0f + exp(-value));;
}

}