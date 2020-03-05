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
    if (type == "MIN") {
        *iou = inter_area / MIN(rect1.area(), rect2.area());
    }
    else {
        *iou = inter_area / (rect1.area() + rect2.area() - inter_area);
    }

    return 0;
}

int GenerateAnchors(const int & width, const int & height,
	const std::vector<std::vector<float>>& min_boxes, const std::vector<float>& strides,
	std::vector<std::vector<float>>* anchors) {
	std::cout << "start generate priors." << std::endl;
	anchors->clear();
	int num_strides = static_cast<int>(strides.size());
	for (int i = 0; i < num_strides; ++i) {
		auto stride = strides[i];
		float scale_x = width / stride;
		float scale_y = height / stride;

		int num_x = ceil(width / stride);
		int num_y = ceil(height / stride);
		for (int y = 0; y < num_y; ++y) {
			for (int x = 0; x < num_x; ++x) {
				float center_x = (x + 0.5f) / scale_x;
				float center_y = (y + 0.5f) / scale_y;
				for (auto min_box : min_boxes[i]) {
					float center_w = min_box / width;
					float center_h = min_box / height;
					anchors->push_back({ Clip(center_x, 1.0f), Clip(center_y, 1.0f),
						Clip(center_w, 1.0f), Clip(center_h, 1.0f) });
				}
			}
		}
	}

	std::cout << "end generate priors." << std::endl;

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