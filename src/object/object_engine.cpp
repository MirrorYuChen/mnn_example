#include "object_engine.h"

#include <iostream>
#include <string>

namespace mirror {
ObjectEngine::ObjectEngine() {
    mobilenetssd_ = new MobilenetSSD();
    initialized_ = false;
}

ObjectEngine::~ObjectEngine() {
    if (mobilenetssd_) {
        delete mobilenetssd_;
        mobilenetssd_ = nullptr;
    }
}

int ObjectEngine::Init(const char* model_path) {
    std::cout << "start init." << std::endl;
    if (mobilenetssd_->Init(model_path) != 0) {
        return 10000;
    }

    initialized_ = true;
    std::cout << "end init." << std::endl;
    return 0;
}

int ObjectEngine::DetectObject(const cv::Mat& img_src, std::vector<ObjectInfo>* objects) {
    return mobilenetssd_->DetectObject(img_src, objects);
}


}