#include "classifier.h"
#include <algorithm>
#include <iostream>

#include "opencv2/imgproc.hpp"

namespace mirror {

Classifier::Classifier() {
    labels_.clear();
    initialized_ = false;
    topk_ = 5;
}

Classifier::~Classifier() {
    mobilenet_interpreter_->releaseModel();
    mobilenet_interpreter_->releaseSession(mobilenet_sess_);
}

int Classifier::Init(const char* root_path) {
    std::cout << "start Init." << std::endl;
    std::string model_file = std::string(root_path) + "/mobilenet.mnn";
    mobilenet_interpreter_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_file.c_str()));
    
    if (!mobilenet_interpreter_ || LoadLabels(root_path) != 0) {
        std::cout << "load model failed." << std::endl;
        return 10000;
    }    
    
    MNN::ScheduleConfig schedule_config;
    schedule_config.type = MNN_FORWARD_CPU;
    schedule_config.numThread = 1;
    MNN::BackendConfig backend_config;
    backend_config.precision = MNN::BackendConfig::Precision_Normal;
    schedule_config.backendConfig = &backend_config;

    mobilenet_sess_ = mobilenet_interpreter_->createSession(schedule_config);
    input_tensor_ = mobilenet_interpreter_->getSessionInput(mobilenet_sess_, nullptr);

    mobilenet_interpreter_->resizeTensor(input_tensor_, {1, 3, inputSize_.height, inputSize_.width});
    mobilenet_interpreter_->resizeSession(mobilenet_sess_);

    std::cout << "End Init." << std::endl; 
    
    initialized_ = true;

    return 0;
}

int Classifier::Classify(const cv::Mat& img_src, std::vector<ImageInfo>* images) {
    std::cout << "start classify." << std::endl;
    images->clear();
    if (!initialized_) {
        std::cout << "model uninitialized." << std::endl;
        return 10000;
    }

    if (img_src.empty()) {
        std::cout << "input empty." << std::endl;
        return 10001;
    }

    cv::Mat img_resized;
    cv::resize(img_src.clone(), img_resized, inputSize_);
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(
        MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, meanVals, 3, normVals, 3)
    );
    pretreat->convert((uint8_t*)img_resized.data, inputSize_.width, inputSize_.height, img_resized.step[0], input_tensor_);

    // forward
    mobilenet_interpreter_->runSession(mobilenet_sess_);

    // get output
    MNN::Tensor* tensor_score = mobilenet_interpreter_->getSessionOutput(mobilenet_sess_, "MobilenetV1/Predictions/Reshape_1");

    std::vector<std::pair<float, int>> scores;
    for (int i = 0; i < 1000; ++i) {
        float score = tensor_score->host<float>()[i];
        std::cout << "score: " << score << std::endl;
        scores.push_back(std::make_pair(score, i));
    }

    std::partial_sort(scores.begin(), scores.begin() + topk_, scores.end(), std::greater< std::pair<float, int> >());
    for (int i = 0; i < topk_; ++i) {
        ImageInfo image_info;
        image_info.label_ = labels_[scores[i].second];
        image_info.score_ = scores[i].first;
        images->push_back(image_info);
    }

    std::cout << "end classify." << std::endl;

    return 0;
}


int Classifier::LoadLabels(const char* root_path) {
    std::string label_file = std::string(root_path) + "/label.txt";
		FILE* fp = fopen(label_file.c_str(), "r");
		while (!feof(fp)) {
			char str[1024];
			fgets(str, 1024, fp);
			std::string str_s(str);

			if (str_s.length() > 0) {
				for (int i = 0; i < str_s.length(); i++) {
					if (str_s[i] == ' ') {
						std::string strr = str_s.substr(i, str_s.length() - i - 1);
						labels_.push_back(strr);
						i = str_s.length();
					}
				}
			}
		}
		return 0;
}



}