#include "mobilenetssd.h"
#include <iostream>
#include <string>

#include "opencv2/imgproc.hpp"

namespace mirror {
MobilenetSSD::MobilenetSSD() {
	initialized_ = false;
}

MobilenetSSD::~MobilenetSSD() {
	mobilenetssd_interpreter_->releaseModel();
	mobilenetssd_interpreter_->releaseSession(mobilenetssd_sess_);
}

int MobilenetSSD::Init(const char * root_path) {
	std::cout << "start Init." << std::endl;
	std::string model_file = std::string(root_path) + "/mobilenetssd.mnn";
	mobilenetssd_interpreter_ = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_file.c_str()));
	if (nullptr == mobilenetssd_interpreter_) {
		std::cout << "load model failed." << std::endl;
		return 10000;
	}

	MNN::ScheduleConfig schedule_config;
	schedule_config.type = MNN_FORWARD_CPU;
	schedule_config.numThread = 4;

	MNN::BackendConfig backend_config;
	backend_config.precision = MNN::BackendConfig::Precision_High;
	backend_config.power = MNN::BackendConfig::Power_High;
	schedule_config.backendConfig = &backend_config;

	mobilenetssd_sess_ = mobilenetssd_interpreter_->createSession(schedule_config);

	// image processer
	MNN::CV::Matrix trans;
	trans.setScale(1.0f, 1.0f);
	MNN::CV::ImageProcess::Config img_config;
	img_config.filterType = MNN::CV::BICUBIC;
	::memcpy(img_config.mean, meanVals_, sizeof(meanVals_));
	::memcpy(img_config.normal, normVals_, sizeof(normVals_));
	img_config.sourceFormat = MNN::CV::BGR;
	img_config.destFormat = MNN::CV::RGB;
	pretreat_data_ = std::shared_ptr<MNN::CV::ImageProcess>(MNN::CV::ImageProcess::create(img_config));
	pretreat_data_->setMatrix(trans);

	std::string input_name = "data";
	input_tensor_ = mobilenetssd_interpreter_->getSessionInput(mobilenetssd_sess_, input_name.c_str());
	mobilenetssd_interpreter_->resizeTensor(input_tensor_, dims_);
	mobilenetssd_interpreter_->resizeSession(mobilenetssd_sess_);

	initialized_ = true;

	std::cout << "end Init." << std::endl;
	return 0;
}

int MobilenetSSD::DetectObject(const cv::Mat & img_src, std::vector<ObjectInfo>* objects) {
	std::cout << "start detect." << std::endl;
	if (!initialized_) {
		std::cout << "model uninitialized." << std::endl;
		return 10000;
	}
	if (img_src.empty()) {
		std::cout << "input empty." << std::endl;
		return 10001;
	}

	int width = img_src.cols;
	int height = img_src.rows;

	// preprocess
	cv::Mat img_resized;
	cv::resize(img_src, img_resized, inputSize_);
	pretreat_data_->convert(img_resized.data, inputSize_.width, inputSize_.height, 0, input_tensor_);
	
	mobilenetssd_interpreter_->runSession(mobilenetssd_sess_);
	std::string output_name = "detection_out";
	MNN::Tensor* output_tensor = mobilenetssd_interpreter_->getSessionOutput(mobilenetssd_sess_, output_name.c_str());

	// copy to host
	MNN::Tensor output_host(output_tensor, output_tensor->getDimensionType());
	output_tensor->copyToHostTensor(&output_host);

	auto output_ptr = output_host.host<float>();
	std::vector<ObjectInfo> objects_tmp;
	for (int i = 0; i < output_host.height(); ++i) {
		int index = i * output_host.width();
		ObjectInfo object;
		object.name_ = class_names[int(output_ptr[index + 0])];
		object.score_ = output_ptr[index + 1];
		object.location_.x = output_ptr[index + 2] * width;
		object.location_.y = output_ptr[index + 3] * height;
		object.location_.width = output_ptr[index + 4] * width - object.location_.x;
		object.location_.height = output_ptr[index + 5] * height - object.location_.y;

		objects_tmp.push_back(object);
	}
	NMS(objects_tmp, objects, nmsThreshold_);

	std::cout << "end detect." << std::endl;

	return 0;
}

}
