#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "classifier.h"

int main(int argc, char* argv[]) {
const char* img_path = "../data/images/cat.jpg";
	cv::Mat img_src = cv::imread(img_path);

	const char* root_path = "../data/models";
	mirror::Classifier* classifier = new mirror::Classifier();

	classifier->Init(root_path);
	std::vector<mirror::ImageInfo> images;
	classifier->Classify(img_src, &images);

	int topk = images.size();
	for (int i = 0; i < topk; ++i) {
		cv::putText(img_src, images[i].label_, cv::Point(10, 10 + 30 * i),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 100, 0), 2, 2);
	}

	cv::imwrite("../data/images/classify_result.jpg", img_src);

    return 0;
}