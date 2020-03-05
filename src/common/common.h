#ifndef _VISION_COMMON_H_
#define _VISION_COMMON_H_

#include <cmath>
#include <string>
#include <vector>
#include "opencv2/core.hpp"

namespace mirror {
#define kFaceFeatureDim 128
#define kFaceNameDim 256
#define Clip(x, y) (x < 0 ? 0 : (x > y ? y : x))
const int threads_num = 2;

struct ImageInfo {
    std::string label_;
    float score_;
};

struct ObjectInfo {
	std::string name_;
	cv::Rect location_;
	float score_;
};

struct FaceInfo {
	cv::Rect location_;
	float score_;
	float keypoints_[10];
};

struct QueryResult {
    std::string name_;
    float sim_;
};

uint8_t* GetImage(const cv::Mat& img_src);
float InterRectArea(const cv::Rect& a, const cv::Rect& b);
int ComputeIOU(const cv::Rect& rect1, const cv::Rect& rect2, float* iou, const std::string& type = "UNION");
int GenerateAnchors(const int& width, const int& height,  const std::vector<std::vector<float>>& min_boxes,
    const std::vector<float>& strides, std::vector<std::vector<float>>* anchors);

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
        int index_good = indexes[0];
        std::vector<int> indexes_tmp = indexes;
        indexes.clear();
        std::vector<int> indexes_nms;
        indexes_nms.push_back(index_good);
        float total = exp(inputs_tmp[index_good].score_);
        for (int i = 1; i < static_cast<int>(indexes_tmp.size()); ++i) {
            int index_tmp = indexes_tmp[i];
            float iou = 0.0f;
            ComputeIOU(inputs_tmp[index_good].location_, inputs_tmp[index_tmp].location_, &iou, type);
            if (iou <= threshold) {
                indexes.push_back(index_tmp);
            } else {
                indexes_nms.push_back(index_tmp);
                total += exp(inputs_tmp[index_tmp].score_);
            }
        }
        if ("BLENDING" == type) {
            T t;
            memset(&t, 0, sizeof(t));
            for (auto index : indexes_nms) {
                float rate = exp(inputs_tmp[index].score_) / total;
                t.score_ += rate * inputs_tmp[index].score_;
				t.location_.x += rate * inputs_tmp[index].location_.x;
				t.location_.y += rate * inputs_tmp[index].location_.y;
				t.location_.width += rate * inputs_tmp[index].location_.width;
				t.location_.height += rate * inputs_tmp[index].location_.height;
            }
            result->push_back(t);
        } else {
            result->push_back(inputs_tmp[index_good]);
        }

    }
}

float CalculateSimilarity(const std::vector<float>&feat1, const std::vector<float>& feat2);
float Logists(const float& value);

}



#endif // !_VISION_COMMON_H_
