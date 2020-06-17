#ifndef _FACE_MOBILEFACENET_H_
#define _FACE_MOBILEFACENET_H_

#include <memory>

#include "../recognizer.h"
#include "MNN/Interpreter.hpp"
#include "MNN/ImageProcess.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"


namespace mirror {
class Mobilefacenet : public Recognizer {
public:
    Mobilefacenet();
    ~Mobilefacenet();
    int Init(const char* model_path);
    int ExtractFeature(const cv::Mat& img_face, std::vector<float>* feat);

private:
    bool initialized_;
    std::shared_ptr<MNN::Interpreter> mobilefacenet_interpreter_ = nullptr;
    MNN::Session* mobilefacenet_sess_ = nullptr;
    MNN::Tensor* input_tensor_ = nullptr;

    const cv::Size inputSize_ = cv::Size(112, 112);
    const float meanVals_[3] = {     127.5f,     127.5f,     127.5f };
    const float normVals_[3] = { 0.0078125f, 0.0078125f, 0.0078125f };
};


}


#endif // !_FACE_MOBILEFACENET_H_
