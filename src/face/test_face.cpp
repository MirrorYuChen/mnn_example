#include "../vision_engine.h"
#include "opencv2/opencv.hpp"

int TestLandmarker(int argc, char* argv[]) {
    const char* img_file = "../../data/images/face.jpg";
    cv::Mat img_src = cv::imread(img_file);

    mirror::VisionEngine* vision_engine = new mirror::VisionEngine();
    const char* root_path = "../../data/models";
    vision_engine->Init(root_path);
    std::vector<mirror::FaceInfo> faces;
    vision_engine->DetectFace(img_src, &faces);
    int num_faces = static_cast<int>(faces.size());
    for (int i = 0; i < num_faces; ++i) {
#if 0
		for (int j = 0; j < 5; ++j) {
			cv::Point curr_pt = cv::Point(faces[i].keypoints_[2 * j], faces[i].keypoints_[2 * j + 1]);
			cv::circle(img_src, curr_pt, 2, cv::Scalar(255, 0, 255), 2);
		}
#endif
        cv::Rect face = faces[i].location_;
        std::vector<cv::Point2f> keypoints;
        vision_engine->ExtractKeypoints(img_src, face, &keypoints);
        cv::Mat face_aligned;
        vision_engine->AlignFace(img_src, keypoints, &face_aligned);
        std::string face_name = "../../data/images/face" + std::to_string(i) + ".jpg";
        cv::imwrite(face_name.c_str(), face_aligned);

        cv::rectangle(img_src, face, cv::Scalar(0, 255, 0), 2);
        int num_keypoints = static_cast<int>(keypoints.size());
        for (int j = 0; j < num_keypoints; ++j) {
            cv::circle(img_src, keypoints[j], 1, cv::Scalar(255, 255, 0), 1);
        }
    }

    cv::imwrite("../../data/images/face_result.jpg", img_src);
	cv::imshow("result", img_src);
	cv::waitKey(0);

    delete vision_engine;

    return 0;
}

int TestRecognizer(int argc, char* argv[]) {
    const char* face1_path = "../../data/images/bb_face1.jpg";
    const char* face2_path = "../../data/images/bb_face2.jpg";
    cv::Mat face1 = cv::imread(face1_path);
    cv::Mat face2 = cv::imread(face2_path);
    
    const char* model_path = "../../data/models";
    mirror::VisionEngine* vision_engine = new mirror::VisionEngine();
    vision_engine->Init(model_path);
    std::vector<float> feat1, feat2;
    vision_engine->ExtractFeature(face1, &feat1);
    vision_engine->ExtractFeature(face2, &feat2);
    float sim = mirror::CalculateSimilarity(feat1, feat2);
    std::cout << "similarity is: " << sim << std::endl;

    return 0;
}

int TestAligner(int argc, char* argv[]) {
    std::string img_path = "../../data/images/" + std::string(argv[1]);
    cv::Mat img_src = cv::imread(img_path);

    mirror::VisionEngine* vision_engine = new mirror::VisionEngine();
    const char* root_path = "../../data/models";
    vision_engine->Init(root_path);
    std::vector<mirror::FaceInfo> faces;
    vision_engine->DetectFace(img_src, &faces);
    int num_faces = static_cast<int>(faces.size());
    for (int i = 0; i < num_faces; ++i) {
        cv::Rect face = faces[i].location_;
        std::vector<cv::Point2f> keypoints;
        vision_engine->ExtractKeypoints(img_src, face, &keypoints);
        cv::Mat face_aligned;
        vision_engine->AlignFace(img_src, keypoints, &face_aligned);
        std::string face_name = "../../data/images/face" + std::to_string(i) + ".jpg";
        cv::imwrite(face_name.c_str(), face_aligned);

        cv::rectangle(img_src, face, cv::Scalar(0, 255, 0), 2);
        int num_keypoints = static_cast<int>(keypoints.size());
        for (int j = 0; j < num_keypoints; ++j) {
            cv::circle(img_src, keypoints[j], 1, cv::Scalar(255, 255, 0), 1);
        }
    }
	cv::imshow("result", img_src);
	cv::waitKey(0);

    delete vision_engine;
}

int TestDatabase(int argc, char* argv[]) {
    const char* img_path = "../../data/images/face.jpg";
    cv::Mat img_src = cv::imread(img_path);
    if (img_src.empty()) {
        std::cout << "load image failed." << std::endl;
        return 10001;
    }

    const char* root_path = "../../data/models";
    mirror::VisionEngine* vision_engine = new mirror::VisionEngine();
    vision_engine->Init(root_path);
    vision_engine->Load();
    std::vector<mirror::FaceInfo> faces;
    vision_engine->DetectFace(img_src, &faces);
    int faces_num = static_cast<int>(faces.size());
    std::cout << "faces number: " << faces_num << std::endl;
    for (int i = 0; i < faces_num; ++i) {
        cv::Rect face = faces.at(i).location_;
		std::vector<cv::Point2f> keypoints;
		vision_engine->ExtractKeypoints(img_src, face, &keypoints);
        cv::Mat face_aligned;
        vision_engine->AlignFace(img_src, keypoints, &face_aligned);
        std::vector<float> feat;
        vision_engine->ExtractFeature(face_aligned, &feat);

#if 0
        vision_engine->Insert(feat, "face" + std::to_string(i));
#endif

#if 1
        mirror::QueryResult query_result;
        vision_engine->QueryTop(feat, &query_result);
        std::cout << i << "-th face is: " << query_result.name_ <<
            " similarity is: " << query_result.sim_ << std::endl;   
#endif
		cv::rectangle(img_src, faces.at(i).location_, cv::Scalar(0, 255, 0), 2);
        cv::Point2f offset = cv::Point2f(faces[i].location_.tl());
		for (int j = 0; j < static_cast<int>(keypoints.size()); ++j) {
			cv::circle(img_src, keypoints[j] + offset, 1, cv::Scalar(0, 0, 255), 1);
		}
    }
    vision_engine->Delete("face0");
    vision_engine->Save();
    cv::imwrite("../../data/images/result.jpg", img_src);

    return 0;
}

int main(int argc, char* argv[]) {
    // return TestRecognizer(argc, argv);
    // return ExtractFace(argc, argv);
    // return TestLandmarker(argc, argv);
    // return TestDatabase(argc, argv);
    return TestAligner(argc, argv);
}