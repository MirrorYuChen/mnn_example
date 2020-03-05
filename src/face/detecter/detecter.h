#ifndef _FACE_DETECTER_H_
#define _FACE_DETECTER_H_

#include "../../common/common.h"

namespace mirror {
class Detecter {
public:
    virtual int Init(const char* model_path) = 0;
	virtual int DetectFace(const cv::Mat& img_src,
        std::vector<FaceInfo>* faces) = 0;
    virtual ~Detecter() {}
};

class DetecterFactory {
public:
    virtual Detecter* CreateDetecter() = 0;
    virtual ~DetecterFactory() {}
};

class CenterfaceFactory : public DetecterFactory {
public:
    CenterfaceFactory() {}
    Detecter* CreateDetecter();
    ~CenterfaceFactory() {}
};

class UltrafaceFactory : public DetecterFactory {
public:
    UltrafaceFactory() {}
    Detecter* CreateDetecter();
    ~UltrafaceFactory() {}
};

}


#endif // !_FACE_DETECTER_H_