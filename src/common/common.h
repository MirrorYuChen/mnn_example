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


}



#endif // !_VISION_COMMON_H_
