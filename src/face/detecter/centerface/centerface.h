#ifndef _FACE_CENTERFACE_H_
#define _FACE_CENTERFACE_H_

#include <vector>
#include <memory>

#include "MNN/Interpreter.hpp"
#include "MNN/ImageProcess.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "../detecter.h"
#include "../../common/common.h"

namespace mirror {
class Centerface : public Detecter {
public:
	Centerface();
	~Centerface();
	int Init(const char* model_path);
	int DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces);

private:
	bool initialized_;
	std::shared_ptr<MNN::CV::ImageProcess> pretreat_ = nullptr;
	std::shared_ptr<MNN::Interpreter> centerface_interpreter_ = nullptr;
	MNN::Session* centerface_sess_ = nullptr;
	MNN::Tensor* input_tensor_ = nullptr;

	const float meanVals_[3] = { 0.0f, 0.0f, 0.0f };
	const float normVals_[3] = { 1.0f, 1.0f, 1.0f };
	const float scoreThreshold_ = 0.5f;
	const float nmsThreshold_ = 0.5f;

};

}


#endif  // !_FACE_CENTERFACE_H_
