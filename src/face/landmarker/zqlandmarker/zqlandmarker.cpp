#include "zqlandmarker.h"

#include <iostream>
#include <string>

#include "opencv2/imgproc.hpp"

namespace mirror {
ZQLandmarker::ZQLandmarker() {
    initialized_ = false;
}

ZQLandmarker::~ZQLandmarker() {
    zq_interpreter_->releaseModel();
    zq_interpreter_->releaseSession(zq_sess_);
}

int ZQLandmarker::Init(const char* model_path) {
    std::cout << "start init." << std::endl;
    std::string model_file = std::string(model_path) + "/zqlandmark.mnn";
    zq_interpreter_ = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_file.c_str()));
    if (nullptr == zq_interpreter_) {
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
    zq_sess_ = zq_interpreter_->createSession(schedule_config);
    input_tensor_ = zq_interpreter_->getSessionInput(zq_sess_, nullptr);

    MNN::CV::Matrix trans;
	trans.setScale(1.0f, 1.0f);
	MNN::CV::ImageProcess::Config img_config;
	img_config.filterType = MNN::CV::BICUBIC;
	::memcpy(img_config.mean, meanVals_, sizeof(meanVals_));
	::memcpy(img_config.normal, normVals_, sizeof(normVals_));
	img_config.sourceFormat = MNN::CV::RGBA;
	img_config.destFormat = MNN::CV::RGB;
	pretreat_ = std::shared_ptr<MNN::CV::ImageProcess>(MNN::CV::ImageProcess::create(img_config));
	pretreat_->setMatrix(trans);

    initialized_ = true;

    std::cout << "end init." << std::endl;
    
    return 0; 
}

int ZQLandmarker::ExtractKeypoints(const cv::Mat& img_src, const cv::Rect& face, std::vector<cv::Point2f>* keypoints) {
    std::cout << "start extract keypoints." << std::endl;
    keypoints->clear();
    if (!initialized_) {
        std::cout << "model uninitialized." << std::endl;
        return 10000;
    }
    if (img_src.empty()) {
        std::cout << "input empty." << std::endl;
        return 10001;
    }
    cv::Mat img_face = img_src(face).clone();
    int width = img_face.cols;
    int height = img_face.rows;
    cv::Mat img_resized;
    cv::resize(img_face, img_resized, inputSize_);
    uint8_t* data_ptr = GetImage(img_resized);
	pretreat_->convert(data_ptr, inputSize_.width, inputSize_.height, 0, input_tensor_);

    // run session
    zq_interpreter_->runSession(zq_sess_);

    // get output
    std::string output_name = "conv6-3";
    auto output_landmark = zq_interpreter_->getSessionOutput(zq_sess_, output_name.c_str());
    MNN::Tensor landmark_tensor(output_landmark, output_landmark->getDimensionType());
    output_landmark->copyToHostTensor(&landmark_tensor);

    for (int i = 0; i < 106; ++i) {
        cv::Point2f curr_pt(landmark_tensor.host<float>()[2 * i + 0] * width + face.x,
                            landmark_tensor.host<float>()[2 * i + 1] * height + face.y);
        keypoints->push_back(curr_pt);
    }

    std::cout << "end extract keypoints." << std::endl;

    return 0;
}


}