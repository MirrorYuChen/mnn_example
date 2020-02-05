#include "common.h"
#include "opencv2/imgproc.hpp"

namespace mirror {

uint8_t* GetImage(const cv::Mat& img_src) {
    uchar* data_ptr = new uchar[img_src.total() * 4];
    cv::Mat img_tmp(img_src.size(), CV_8UC4, data_ptr);
    cv::cvtColor(img_src, img_tmp, cv::COLOR_BGR2RGBA, 4);
    return (uint8_t*)img_tmp.data;
}


}