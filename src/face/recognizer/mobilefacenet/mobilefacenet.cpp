#include "mobilefacenet.h"

#include <iostream>
#include <string>

#include "opencv2/imgproc.hpp"

namespace mirror {
Mobilefacenet::Mobilefacenet() {
    initialized_ = false;
}

Mobilefacenet::~Mobilefacenet() {
    mobilefacenet_interpreter_->releaseModel();
    mobilefacenet_interpreter_->releaseSession(mobilefacenet_sess_);
}

int Mobilefacenet::Init(const char* model_path) {
    std::cout << "start init." << std::endl;
    std::string model_file = std::string(model_path) + "/mobilefacenet.mnn";
    mobilefacenet_interpreter_ = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_file.c_str()));
    if (nullptr == mobilefacenet_interpreter_) {
        std::cout << "load model failed." << std::endl;
        return 10000;
    }

    // create session
    MNN::ScheduleConfig schedule_config;
    schedule_config.type = MNN_FORWARD_CPU;
    schedule_config.numThread = 1;
    MNN::BackendConfig backend_config;
    backend_config.memory    = MNN::BackendConfig::Memory_Normal;
    backend_config.power     = MNN::BackendConfig::Power_Normal;
    backend_config.precision = MNN::BackendConfig::Precision_Normal;
    schedule_config.backendConfig = &backend_config;
    mobilefacenet_sess_ = mobilefacenet_interpreter_->createSession(schedule_config);
    input_tensor_ = mobilefacenet_interpreter_->getSessionInput(mobilefacenet_sess_, nullptr);

    initialized_ = true;

    std::cout << "end init." << std::endl;

    return 0; 
}

int Mobilefacenet::ExtractFeature(const cv::Mat& img_face, std::vector<float>* feat) {
    std::cout << "start extract feature." << std::endl;
    feat->clear();
    if (!initialized_) {
        std::cout << "model uninitialized." << std::endl;
        return 10000;
    }
    if (img_face.empty()) {
        std::cout << "input empty." << std::endl;
        return 10001;
    }

    cv::Mat img_resized;
    cv::resize(img_face, img_resized, inputSize_);
    img_resized.convertTo(img_resized, CV_32FC3);

    // MNN ImageProcess不支持dstformat为MNN_DATA_FORMAT_NCHW
    // 需要自己定义TENSORFLOW或CAFFE_C4格式tensor进行赋值
    auto nhwc_tensor = MNN::Tensor::create<float>({ 1,inputSize_.height, inputSize_.width, 3 }, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data   = nhwc_tensor->host<float>();
    auto nhwc_size   = nhwc_tensor->size();
    ::memcpy(nhwc_data, img_resized.data, nhwc_size);
    input_tensor_->copyFromHostTensor(nhwc_tensor);
    delete nhwc_tensor;

    mobilefacenet_interpreter_->runSession(mobilefacenet_sess_);

    // get output
    std::string output_name = "fc1";
    auto output_feat = mobilefacenet_interpreter_->getSessionOutput(mobilefacenet_sess_, output_name.c_str());
    MNN::Tensor feat_tensor(output_feat, output_feat->getDimensionType());
    output_feat->copyToHostTensor(&feat_tensor);

    std::cout << "channel: " << feat_tensor.channel() << std::endl
        << "width: " << feat_tensor.width() << std::endl
        << "height: " << feat_tensor.height() << std::endl;
    feat->resize(kFaceFeatureDim);
    for (int i = 0; i < kFaceFeatureDim; ++i) {
        feat->at(i) = feat_tensor.host<float>()[i];
    }

    std::cout << "end extract feature." << std::endl;
    return 0;
}




}