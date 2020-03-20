#ifndef _FACE_LANDMARKER_H_
#define _FACE_LANDMARKER_H_

#include "../../common/common.h"

namespace mirror {
class Landmarker {
public:
    virtual int Init(const char* model_path) = 0;
    virtual int ExtractKeypoints(const cv::Mat& img_src, const cv::Rect& face, std::vector<cv::Point2f>* keypoints) = 0;
    virtual ~Landmarker() {}
};

class LandmarkerFactory {
public:
    virtual Landmarker* CreateLandmarker() = 0;
    virtual ~LandmarkerFactory() {}
};

class PFLDLandmarkerFactory : public LandmarkerFactory {
public:
    PFLDLandmarkerFactory() {}
    Landmarker* CreateLandmarker();
    ~PFLDLandmarkerFactory() {}
};

class ZQLandmarkerFactory : public LandmarkerFactory {
public:
    ZQLandmarkerFactory() {}
    Landmarker* CreateLandmarker();
    ~ZQLandmarkerFactory() {}
};




}





#endif // !_FACE_LANDMARKER_H_