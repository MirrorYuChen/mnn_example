#include "vision_engine.h"
#include <iostream>

#include "classifier/classifier.h"
#include "object/object_engine.h"
#include "face/face_engine.h"
#include "object/object_engine.h"


namespace mirror {
class VisionEngine::Impl {
public:
    Impl() {
        face_engine_ = new FaceEngine();
        object_engine_ = new ObjectEngine();
        classifier_ = new Classifier();
        initialized_ = false;
    }

    ~Impl() {
        if (classifier_) {
            delete classifier_;
            classifier_ = nullptr;
        }

        if (face_engine_) {
            delete face_engine_;
            face_engine_ = nullptr;
        }
        if (object_engine_) {
            delete object_engine_;
            object_engine_ = nullptr;
        }
    }

    int Init(const char* root_path) {
        if (classifier_->Init(root_path) != 0) {
            std::cout << "init classifier failed." << std::endl;
            return 10000;
        }

        if (object_engine_->Init(root_path) != 0) {
            std::cout << "init object detector failed." << std::endl;
            return 10000;
        }

        if (face_engine_->Init(root_path) != 0) {
            std::cout << "Init face engine failed." << std::endl;
            return 10000;
        }

        initialized_ = true;

        return 0;
    }
    inline int Classify(const cv::Mat& img_src, std::vector<ImageInfo>* images) {        
        return classifier_->Classify(img_src, images);
    }
    inline int DetectObject(const cv::Mat& img_src, std::vector<ObjectInfo>* objects) {        
        return object_engine_->DetectObject(img_src, objects);
    }

    inline int DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces) {
        return face_engine_->DetectFace(img_src, faces);
    }
    inline int ExtractKeypoints(const cv::Mat& img_src, const cv::Rect& face, std::vector<cv::Point2f>* keypoints) {
        return face_engine_->ExtractKeypoints(img_src, face, keypoints);
    }
    inline int AlignFace(const cv::Mat& img_src, const std::vector<cv::Point2f>& keypoints, cv::Mat* face_aligned) {
        return face_engine_->AlignFace(img_src, keypoints, face_aligned);
    }
    inline int ExtractFeature(const cv::Mat& img_face, std::vector<float>* feat) {
        return face_engine_->ExtractFeature(img_face, feat);
    }

    inline int Insert(const std::vector<float>& feat, const std::string& name) {
        return face_engine_->Insert(feat, name);
    }
    inline int Delete(const std::string& name) {
        return face_engine_->Delete(name);
    }
	inline int64_t QueryTop(const std::vector<float>& feat, QueryResult *query_result = nullptr) {
        return face_engine_->QueryTop(feat, query_result);
    }
    inline int Save() {
        return  face_engine_->Save();
    }
    inline int Load() {
        return face_engine_->Load();
    }

private:
    bool initialized_;
    FaceEngine* face_engine_;
    ObjectEngine* object_engine_;
    Classifier* classifier_;

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

int VisionEngine::DetectObject(const cv::Mat& img_src, std::vector<ObjectInfo>* objects) {
    return impl_->DetectObject(img_src, objects);
}

int VisionEngine::DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces) {
    return impl_->DetectFace(img_src, faces);
}

int VisionEngine::ExtractKeypoints(const cv::Mat& img_src, const cv::Rect& face, std::vector<cv::Point2f>* keypoints) {
    return impl_->ExtractKeypoints(img_src, face, keypoints);
}

int VisionEngine::AlignFace(const cv::Mat& img_src, const std::vector<cv::Point2f>& keypoints, cv::Mat* face_aligned) {
    return impl_->AlignFace(img_src, keypoints, face_aligned);
}

int VisionEngine::ExtractFeature(const cv::Mat& img_face, std::vector<float>* feat) {
    return impl_->ExtractFeature(img_face, feat);
}

int VisionEngine::Insert(const std::vector<float>& feat, const std::string& name) {
    return impl_->Insert(feat, name);
}

int VisionEngine::Delete(const std::string& name) {
    return impl_->Delete(name);
}

int64_t VisionEngine::QueryTop(const std::vector<float>& feat,
    QueryResult* query_result) {
    return impl_->QueryTop(feat, query_result);
}

int VisionEngine::Save() {
    return impl_->Save();
}

int VisionEngine::Load() {
    return impl_->Load();
}

}