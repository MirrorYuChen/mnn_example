#ifndef _FACE_ENGINE_H_
#define _FACE_ENGINE_H_

#include <vector>
#include "../common/common.h"

namespace mirror {
class FaceEngine {
public:
    FaceEngine();
    ~FaceEngine();
    int Init(const char* root_path);
    int DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces);
    int ExtractKeypoints(const cv::Mat& img_src, const cv::Rect& face, std::vector<cv::Point2f>* keypoints);
    int AlignFace(const cv::Mat& img_src, const std::vector<cv::Point2f>& keypoints, cv::Mat* face_aligned);
    int ExtractFeature(const cv::Mat& img_face, std::vector<float>* feat);

    int Insert(const std::vector<float>& feat, const std::string& name);
	int Delete(const std::string& name);
	int64_t QueryTop(const std::vector<float>& feat, QueryResult *query_result = nullptr);
    int Save();
    int Load();

private:
    class Impl;
    Impl* impl_;

};


}




#endif // !_FACE_ENGINE_H_