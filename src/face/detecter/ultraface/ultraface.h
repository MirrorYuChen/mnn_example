#ifndef _FACE_ULTRAFACE_H_
#define _FACE_ULTRAFACE_H_

#include <vector>
#include <memory>
#include "opencv2/core.hpp"

#include "MNN/Interpreter.hpp"
#include "MNN/ImageProcess.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"

#include "../detecter.h"
#include "../../common/common.h"

namespace mirror {
class UltraFace : public Detecter {
public:
	UltraFace();
	~UltraFace();
	int Init(const char* model_path);
	int DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces);

private:
	bool initialized_;
	std::shared_ptr<MNN::CV::ImageProcess> pretreat_ = nullptr;
	std::shared_ptr<MNN::Interpreter> ultraface_interpreter_ = nullptr;
	MNN::Session* ultraface_sess_ = nullptr;
	MNN::Tensor* input_tensor_ = nullptr;

	const cv::Size inputSize_ = { 320, 240 };
	const float meanVals_[3] = {     127.0f,     127.0f,     127.0f };
	const float normVals_[3] = { 0.0078125f, 0.0078125f, 0.0078125f };
	const float centerVariance_ = 0.1f;
	const float sizeVariance_ = 0.2f;
	const std::vector<std::vector<float>> minBoxes_ = {
		{  10.0f,  16.0f, 24.0f  },
		{  32.0f,  48.0f },
		{  64.0f,  96.0f },
		{ 128.0f, 192.0f, 256.0f }
	};
	const std::vector<float> strides_ = { 8.0f, 16.0f, 32.0f, 64.0f };
	std::vector<std::vector<float>> anchors_ = {};

	const float scoreThreshold_ = 0.65f;
	const float iouThreshold_ = 0.3f;
};

}


#endif // !_FACE_ULTRAFACE_H_
