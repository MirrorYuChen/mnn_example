#include "vision_engine.h"
#include <iostream>

#include "classifier/classifier.h"
#include "object/mobilenetssd.h"


namespace mirror {
class VisionEngine::Impl {
public:
    Impl() : classifier_(new Classifier()),
    mobilenetssd_(new MobilenetSSD()),
    initialized_(false) {}

    ~Impl() {
        if (classifier_) {
            delete classifier_;
            classifier_ = nullptr;
        }

        if (mobilenetssd_) {
            delete mobilenetssd_;
            mobilenetssd_ = nullptr;
        }
    }

    int Init(const char* root_path) {
        if (classifier_->Init(root_path) != 0) {
            std::cout << "init classifier failed." << std::endl;
            return 10000;
        }

        if (mobilenetssd_->Init(root_path) != 0) {
            std::cout << "init object detector failed." << std::endl;
            return 10000;
        }

        initialized_ = true;
        return 0;
    }
    int Classify(const cv::Mat& img_src, std::vector<ImageInfo>* images) {        
        return classifier_->Classify(img_src, images);
    }
    int Detect(const cv::Mat& img_src, std::vector<ObjectInfo>* objects) {        
        return mobilenetssd_->Detect(img_src, objects);
    }

private:
    bool initialized_;
    Classifier* classifier_;
    MobilenetSSD* mobilenetssd_;

};

VisionEngine::VisionEngine() {
    impl_ = new VisionEngine::Impl();
}

VisionEngine::~VisionEngine() {
    if (impl_) {
        delete impl_;
        impl_ = nullptr;
    }
}

int VisionEngine::Init(const char* root_path) {
    return impl_->Init(root_path);
}

int VisionEngine::Classify(const cv::Mat& img_src, std::vector<ImageInfo>* images) {
    return impl_->Classify(img_src, images);
}

int VisionEngine::Detect(const cv::Mat& img_src, std::vector<ObjectInfo>* objects) {
    return impl_->Detect(img_src, objects);
}


}