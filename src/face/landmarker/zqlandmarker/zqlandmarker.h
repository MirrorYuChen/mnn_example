#ifndef _FACE_ZQLANDMARKER_H_
#define _FACE_ZQLANDMARKER_H_

#include <vector>

#include "opencv2/core.hpp"
#include "../landmarker.h"

#include "MNN/Interpreter.hpp"
#include "MNN/ImageProcess.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"

namespace mirror {
class ZQLandmarker : public Landmarker {
public:
    ZQLandmarker();
    ~ZQLandmarker();
    int Init(const char* model_path);
    int ExtractKeypoints(const cv::Mat& img_src, const cv::Rect& face, std::vector<cv::Point2f>* keypoints);

private:
    bool initialized_;
    std::shared_ptr<MNN::CV::ImageProcess> pretreat_ = nullptr;
    std::shared_ptr<MNN::Interpreter> zq_interpreter_ = nullptr;
    MNN::Session* zq_sess_ = nullptr;
    MNN::Tensor* input_tensor_ = nullptr;
    
    const cv::Size inputSize_ = cv::Size(112, 112);
    const float meanVals_[3] = {   127.5f, 127.5f, 127.5f };
    const float normVals_[3] = { 0.0078125f, 0.0078125f, 0.0078125f };

};


}




#endif // !_FACE_ZQLANDMARKER_H_