#ifndef _VISION_COMMON_H_
#define _VISION_COMMON_H_

#include <string>
#include <vector>
#include "opencv2/core.hpp"

namespace mirror {

struct ImageInfo {
    std::string label_;
    float score_;
};

struct ObjectInfo {
	std::string name_;
	cv::Rect location_;
	float score_;
};

uint8_t* GetImage(const cv::Mat& img_src);

float InterRectArea(const cv::Rect& a, const cv::Rect& b);
int ComputeIOU(const cv::Rect& rect1, const cv::Rect& rect2, float* iou, const std::string& type = "UNION");

template <typename T>
int const NMS(const std::vector<T>& inputs, std::vector<T>* result,
	const float& threshold, const std::string& type = "UNION") {
	result->clear();
    if (inputs.size() == 0)
        return -1;
    
    std::vector<T> inputs_tmp;
    inputs_tmp.assign(inputs.begin(), inputs.end());
    std::sort(inputs_tmp.begin(), inputs_tmp.end(),
    [](const T& a, const T& b) {
        return a.score_ < b.score_;
    });

    std::vector<int> indexes(inputs_tmp.size());

    for (int i = 0; i < indexes.size(); i++) {
        indexes[i] = i;
    }

    while (indexes.size() > 0) {
        int good_idx = indexes[0];
        result->push_back(inputs_tmp[good_idx]);
        std::vector<int> tmp_indexes = indexes;
        indexes.clear();
        for (int i = 1; i < tmp_indexes.size(); i++) {
            int tmp_i = tmp_indexes[i];
            float iou = 0.0f;
            ComputeIOU(inputs_tmp[good_idx].location_, inputs_tmp[tmp_i].location_, &iou, type);
            if (iou <= threshold) {
                indexes.push_back(tmp_i);
            }
        }
    }
}


}



#endif // !_VISION_COMMON_H_
