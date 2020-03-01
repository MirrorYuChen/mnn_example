#include "../vision_engine.h"
#include "opencv2/opencv.hpp"

int TestObject(int argc, char* argv[]){
	const char* img_path = "../../data/images/object.jpg";
	cv::Mat img_src = cv::imread(img_path);
	mirror::VisionEngine* vision_engine = new mirror::VisionEngine();

	const char* root_path = "../../data/models";
	vision_engine->Init(root_path);
	std::vector<mirror::ObjectInfo> objects;
	vision_engine->DetectObject(img_src, &objects);

	int num_objects = static_cast<int>(objects.size());
	for (int i = 0; i < num_objects; ++i) {
		std::cout << "location: " << objects[i].location_ << std::endl;
		cv::rectangle(img_src, objects[i].location_, cv::Scalar(255, 0, 255), 2);
		char text[256];
		sprintf(text, "%s %.1f%%", objects[i].name_.c_str(), objects[i].score_ * 100);
		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		cv::putText(img_src, text, cv::Point(objects[i].location_.x,
			objects[i].location_.y + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
	}
	cv::imwrite("../../data/images/object_result.jpg", img_src);
	cv::imshow("result", img_src);
	cv::waitKey(0);

	delete vision_engine;
	return 0;
}

int main(int argc, char* argv[]) {
	return TestObject(argc, argv);
}
