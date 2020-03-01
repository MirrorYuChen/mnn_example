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
    classifier_interpreter_->releaseModel();
    classifier_interpreter_->releaseSession(classifier_sess_);
}

int Classifier::Init(const char* root_path) {
    std::cout << "start Init." << std::endl;
    std::string model_file = std::string(root_path) + "/mobilenet.mnn";
    classifier_interpreter_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_file.c_str()));
    
    if (!classifier_interpreter_ || LoadLabels(root_path) != 0) {
        std::cout << "load model failed." << std::endl;
        return 10000;
    }    
    
    MNN::ScheduleConfig schedule_config;
    schedule_config.type = MNN_FORWARD_CPU;
    schedule_config.numThread = 1;
    MNN::BackendConfig backend_config;
    backend_config.precision = MNN::BackendConfig::Precision_Normal;
    schedule_config.backendConfig = &backend_config;

    classifier_sess_ = classifier_interpreter_->createSession(schedule_config);
    input_tensor_ = classifier_interpreter_->getSessionInput(classifier_sess_, nullptr);

    classifier_interpreter_->resizeTensor(input_tensor_, {1, 3, inputSize_.height, inputSize_.width});
    classifier_interpreter_->resizeSession(classifier_sess_);

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
    classifier_interpreter_->runSession(classifier_sess_);

    // get output
    // mobilenet: "classifierV1/Predictions/Reshape_1"
    MNN::Tensor* output_score = classifier_interpreter_->getSessionOutput(classifier_sess_, nullptr);

    // copy to host
    MNN::Tensor score_host(output_score, output_score->getDimensionType());
    output_score->copyToHostTensor(&score_host);

    auto score_ptr = score_host.host<float>();
    std::vector<std::pair<float, int>> scores;
    for (int i = 0; i < 1000; ++i) {
        float score = score_ptr[i];
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
			if (fgets(str, 1024, fp) == nullptr) continue;
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