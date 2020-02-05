#ifndef _VISION_ENGINE_H_
#define _VISION_ENGINE_H_

#include <vector>
#include "common/common.h"

namespace mirror {
class VisionEngine {
public:
    VisionEngine();
    ~VisionEngine();
    int Init(const char* root_path);
    int Classify(const cv::Mat& img_src, std::vector<ImageInfo>* images);
    int Detect(const cv::Mat& img_src, std::vector<ObjectInfo>* objects);

private:
    class Impl;
    Impl* impl_;

};


}




#endif // !_VISION_ENGINE_