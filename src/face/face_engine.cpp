#include "face_engine.h"
#include <iostream>

#include "detecter/detecter.h"
#include "landmarker/landmarker.h"
#include "aligner/aligner.h"
#include "recognizer/recognizer.h"
#include "database/face_database.h"


namespace mirror {
class FaceEngine::Impl {
public:
    Impl() {
        detecter_factory_ = new UltrafaceFactory();
        landmarker_factory_ = new ZQLandmarkerFactory();
        recognizer_factory_ = new MobilefacenetFactory();
        
        detecter_ = detecter_factory_->CreateDetecter();
        landmarker_ = landmarker_factory_->CreateLandmarker();
        recognizer_ = recognizer_factory_->CreateRecognizer();

        aligner_ = new Aligner();
        database_ = new FaceDatabase();
        initialized_ = false;
    }

    ~Impl() {
        if (detecter_) {
            delete detecter_;
            detecter_ = nullptr;
        }

        if (landmarker_) {
            delete landmarker_;
            landmarker_ = nullptr;
        }

        if (recognizer_) {
            delete recognizer_;
            recognizer_ = nullptr;
        }

        if (database_) {
            delete database_;
            database_ = nullptr;
        }

        if (detecter_factory_) {
            delete detecter_factory_;
            detecter_factory_ = nullptr;
        }

        if (landmarker_factory_) {
            delete landmarker_factory_;
            landmarker_factory_ = nullptr;
        }

        if (recognizer_factory_) {
            delete recognizer_factory_;
            recognizer_factory_ = nullptr;
        }
    }

    int Init(const char* root_path) {
        if (detecter_->Init(root_path) != 0) {
            std::cout << "Init face detecter failed." << std::endl;
            return 10000;
        }

        if (landmarker_->Init(root_path) != 0) {
            std::cout << "Init face landmarker failed." << std::endl;
            return 10000;
        }

        if (recognizer_->Init(root_path) != 0) {
            std::cout << "Init face recognizer failed." << std::endl;
            return 10000;
        }

        db_name_ = std::string(root_path);
        initialized_ = true;

        return 0;
    }
    inline int DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces) {
        return detecter_->DetectFace(img_src, faces);
    }
    inline int ExtractKeypoints(const cv::Mat& img_src, const cv::Rect& face, std::vector<cv::Point2f>* keypoints) {
        return landmarker_->ExtractKeypoints(img_src, face, keypoints);
    }
    inline int AlignFace(const cv::Mat& img_src, const std::vector<cv::Point2f>& keypoints, cv::Mat* face_aligned) {
        return aligner_->AlignFace(img_src, keypoints, face_aligned);
    }
    inline int ExtractFeature(const cv::Mat& img_face, std::vector<float>* feat) {
        return recognizer_->ExtractFeature(img_face, feat);
    }

    inline int Insert(const std::vector<float>& feat, const std::string& name) {
        return database_->Insert(feat, name);
    }
    inline int Delete(const std::string& name) {
        return database_->Delete(name);
    }
	inline int64_t QueryTop(const std::vector<float>& feat, QueryResult *query_result = nullptr) {
        return database_->QueryTop(feat, query_result);
    }
    inline int Save() {
        return  database_->Save(db_name_.c_str());
    }
    inline int Load() {
        return database_->Load(db_name_.c_str());
    }

private:
    DetecterFactory* detecter_factory_;
    LandmarkerFactory* landmarker_factory_;
    RecognizerFactory* recognizer_factory_;

private:
    bool initialized_;
    std::string db_name_;
    Aligner* aligner_;
    Detecter* detecter_;
    Landmarker* landmarker_;
    Recognizer* recognizer_;
    FaceDatabase* database_;
};

FaceEngine::FaceEngine() {
    impl_ = new FaceEngine::Impl();
}

FaceEngine::~FaceEngine() {
    if (impl_) {
        delete impl_;
        impl_ = nullptr;
    }
}

int FaceEngine::Init(const char* root_path) {
    return impl_->Init(root_path);
}

int FaceEngine::DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces) {
    return impl_->DetectFace(img_src, faces);
}

int FaceEngine::ExtractKeypoints(const cv::Mat& img_src, const cv::Rect& face, std::vector<cv::Point2f>* keypoints) {
    return impl_->ExtractKeypoints(img_src, face, keypoints);
}

int FaceEngine::AlignFace(const cv::Mat& img_src, const std::vector<cv::Point2f>& keypoints, cv::Mat* face_aligned) {
    return impl_->AlignFace(img_src, keypoints, face_aligned);
}

int FaceEngine::ExtractFeature(const cv::Mat& img_face, std::vector<float>* feat) {
    return impl_->ExtractFeature(img_face, feat);
}

int FaceEngine::Insert(const std::vector<float>& feat, const std::string& name) {
    return impl_->Insert(feat, name);
}

int FaceEngine::Delete(const std::string& name) {
    return impl_->Delete(name);
}

int64_t FaceEngine::QueryTop(const std::vector<float>& feat,
    QueryResult* query_result) {
    return impl_->QueryTop(feat, query_result);
}

int FaceEngine::Save() {
    return impl_->Save();
}

int FaceEngine::Load() {
    return impl_->Load();
}

}