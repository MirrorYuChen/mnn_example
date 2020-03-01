#ifndef _OBJECTER_H_
#define _OBJECTER_H_

#include "../../common/common.h"
#include "./mobilenetssd/mobilenetssd.h"

namespace mirror {
class ObjectEngine {
public:
    ObjectEngine();
    ~ObjectEngine();
    int Init(const char* root_path);
    int DetectObject(const cv::Mat& img_src, std::vector<ObjectInfo>* objects);

private:
    bool initialized_;
    MobilenetSSD* mobilenetssd_;

};

}



#endif // !_OBJECTER_H_
