# mnn_example
alibaba MNN, mobilenet classifier, centerface detector, pfld landmarker and zqlandmarker, mobilefacenet
## 2020.03.20: fix bug in face aligner
## 2020.03.05: add ultraface and blending nms
## 2020.02.21: add mobilefacenet
## 2020.02.20: use template to reduce the reaptly code in NMS
## 2020.02.18: refacter the project
## 2020.02.17: add zwnet and face database
## 2020.02.16: refacter the project and add zqlandmarker
## 2020.02.15: add pfld landmarker and face aligner
## 2020.02.14: add centerface detector
## 2020.02.05: add object detection
## 2020.01.28: add image classification
# **How to use?**
## 1. convert model
## classifier model comes from: https://github.com/tensorflow/models/tree/master/research/slim
## object detection model comes from: https://github.com/C-Aniruddh/realtime_object_recognition
```
./MNNConvert -f TF --modelFile mobilenet_v1_1.0_224_frozen.pb --MNNModel mobilenet.mnn --bizCode MNN
```
```
./MNNConvert -f CAFFE --modelFile MobileNetSSD_deploy.caffemodel --prototxt MobileNetSSD_deploy.prototxt --MNNModel mobilenetssd.mnn --bizCode MNN
```
## 2. build
```
mkdir build && cd build && make -j3 &&  cd src && ./classifier && ./object && ./face
```
## 3. classifier result:
![图片](https://github.com/MirrorYuChen/MNN_mobilenet/blob/master/data/images/classify_result.jpg)
## 4. object result:
![图片](https://github.com/MirrorYuChen/MNN_mobilenet/blob/master/data/images/object_result.jpg)
## 5. face detection result:
![图片](https://github.com/MirrorYuChen/MNN_mobilenet/blob/master/data/images/face_result.jpg)
## 6. TODO:
 - [x] add pose
## 7. reference:
## MNN: https://github.com/alibaba/MNN
## ZQCNN: https://github.com/zuoqing1988/ZQCNN
## MNN_APPLICATION: https://github.com/xindongzhang/MNN-APPLICATIONS
## insightface: https://github.com/deepinsight/insightface
## centerface: https://github.com/Star-Clouds/CenterFace
## ultraface: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
## seetaface2: https://github.com/seetafaceengine/SeetaFace2
## csdn blog: https://blog.csdn.net/abcd740181246/article/details/90143848
