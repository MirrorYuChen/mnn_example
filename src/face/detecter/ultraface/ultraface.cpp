#include "ultraface.h"

#include <algorithm>
#include <iostream>
#include <string>

#include "opencv2/imgproc.hpp"

#define Clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

namespace mirror {
UltraFace::UltraFace() {
	anchors_.clear();
	initialized_ = false;
}

UltraFace::~UltraFace() {
	ultraface_interpreter_->releaseModel();
	ultraface_interpreter_->releaseSession(ultraface_sess_);
}

int UltraFace::Init(const char * model_path) {
	std::cout << "start init ultraface." << std::endl;
	std::string model_file = std::string(model_path) + "/RFB-320.mnn";
	std::cout << "model path: " << model_file << std::endl;
	ultraface_interpreter_ = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_file.c_str()));

	if (nullptr == ultraface_interpreter_) {
		std::cout << "load model failed." << std::endl;
		return 10000;
	}

	MNN::ScheduleConfig schedule_config;
	schedule_config.type = MNN_FORWARD_CPU;
	schedule_config.numThread = 1;
	MNN::BackendConfig bacckend_config;
	bacckend_config.memory = MNN::BackendConfig::Memory_Normal;
	bacckend_config.power = MNN::BackendConfig::Power_Normal;
	bacckend_config.precision = MNN::BackendConfig::Precision_Normal;
	schedule_config.backendConfig = &bacckend_config;

	ultraface_sess_ = ultraface_interpreter_->createSession(schedule_config);
	input_tensor_ = ultraface_interpreter_->getSessionInput(ultraface_sess_, nullptr);
	ultraface_interpreter_->resizeTensor(input_tensor_, {1, 3, inputSize_.height, inputSize_.width});
	ultraface_interpreter_->resizeSession(ultraface_sess_); 

	MNN::CV::Matrix trans;
	trans.setScale(1.0f, 1.0f);
	MNN::CV::ImageProcess::Config img_config;
	img_config.filterType = MNN::CV::BICUBIC;
	::memcpy(img_config.mean, meanVals_, sizeof(meanVals_));
	::memcpy(img_config.normal, normVals_, sizeof(normVals_));
	img_config.sourceFormat = MNN::CV::BGR;
	img_config.destFormat   = MNN::CV::RGB;
	pretreat_ = std::shared_ptr<MNN::CV::ImageProcess>(MNN::CV::ImageProcess::create(img_config));
	pretreat_->setMatrix(trans);

	GenerateAnchors(inputSize_.width, inputSize_.height, minBoxes_, strides_, &anchors_);

	initialized_ = true;


	std::cout << "end init ultraface." << std::endl;
	return 0;
}

int UltraFace::DetectFace(const cv::Mat & img_src, std::vector<FaceInfo>* faces) {
	std::cout << "start detect face." << std::endl;
	faces->clear();
	if (!initialized_) {
		std::cout << "model uninitialized." << std::endl;
		return 10000;
	}
	if (img_src.empty()) {
		std::cout << "model empty." << std::endl;
		return 10001;
	}

	int width = img_src.cols;
	int height = img_src.rows;

	cv::Mat img_resized;
	cv::resize(img_src, img_resized, inputSize_);
	pretreat_->convert(img_resized.data, inputSize_.width, inputSize_.height, img_resized.step[0], input_tensor_);

	ultraface_interpreter_->runSession(ultraface_sess_);

	auto tensor_score = ultraface_interpreter_->getSessionOutput(ultraface_sess_, "scores");
	auto tensor_bbox  = ultraface_interpreter_->getSessionOutput(ultraface_sess_, "boxes");
	MNN::Tensor host_score(tensor_score, tensor_score->getDimensionType());
	MNN::Tensor host_bbox(tensor_bbox, tensor_bbox->getDimensionType());
	tensor_score->copyToHostTensor(&host_score);
	tensor_bbox->copyToHostTensor(&host_bbox);

	int num_anchors = static_cast<int>(anchors_.size());
	std::vector<FaceInfo> faces_tmp;
	for (int i = 0; i < num_anchors; ++i) {
		float score = host_score.host<float>()[2 * i + 1];
		if (score <= scoreThreshold_) continue;
		FaceInfo face_info;
		float center_x = host_bbox.host<float>()[4 * i] * centerVariance_ * anchors_[i][2] + anchors_[i][0];
		float center_y = host_bbox.host<float>()[4 * i + 1] * centerVariance_ * anchors_[i][3] + anchors_[i][1];
		float center_w = exp(host_bbox.host<float>()[4 * i + 2] * sizeVariance_) * anchors_[i][2];
		float center_h = exp(host_bbox.host<float>()[4 * i + 3] * sizeVariance_) * anchors_[i][3];
		cv::Rect face;
		face.x = static_cast<int>(Clip(center_x - center_w / 2.0f, 1.0f) * width);
		face.y = static_cast<int>(Clip(center_y - center_h / 2.0f, 1.0f) * height);
		face.width = static_cast<int>(Clip(center_w, 1.0f) * width);
		face.height = static_cast<int>(Clip(center_h, 1.0f) * height);

        int max_side = MAX(face.width, face.height);
        face_info.location_.x = face.x + 0.5f * face.width - 0.5f * max_side;
		face_info.location_.y = face.y + 0.5f * face.height - 0.5f * max_side;
		face_info.location_.width = max_side;
		face_info.location_.height = max_side;
		face_info.location_ = face_info.location_ & cv::Rect(0, 0, width, height);

		face_info.score_ = Clip(score, 1.0f);
		faces_tmp.push_back(face_info);
	}
	NMS(faces_tmp, faces, iouThreshold_, "BLENDING");

	std::cout << "end detect face." << std::endl;

	return 0;
}




}