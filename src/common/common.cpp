#include "common.h"
#include "opencv2/imgproc.hpp"

namespace mirror {

uint8_t* GetImage(const cv::Mat& img_src) {
    uchar* data_ptr = new uchar[img_src.total() * 4];
    cv::Mat img_tmp(img_src.size(), CV_8UC4, data_ptr);
    cv::cvtColor(img_src, img_tmp, cv::COLOR_BGR2RGBA, 4);
    return (uint8_t*)img_tmp.data;
}

float InterRectArea(const cv::Rect& a, const cv::Rect& b) {
    cv::Point left_top = cv::Point(MAX(a.x, b.x), MAX(a.y, b.y));
    cv::Point right_bottom = cv::Point(MIN(a.br().x, b.br().x), MIN(a.br().y, b.br().y));
    cv::Point diff = right_bottom - left_top;
    return (MAX(diff.x + 1, 0) * MAX(diff.y + 1, 0));
}

int ComputeIOU(const cv::Rect& rect1,
    const cv::Rect& rect2, float* iou,
    const std::string& type) {

    float inter_area = InterRectArea(rect1, rect2);
    if (type == "UNION") {
        *iou = inter_area / (rect1.area() + rect2.area() - inter_area);
    }
    else {
        *iou = inter_area / MIN(rect1.area(), rect2.area());
    }

    return 0;
}

}