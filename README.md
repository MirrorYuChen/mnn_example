# mnn_example
alibaba MNN, classifier: mobilenet, object detection: mobilenetssd 
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
mkdir build && cd build && make -j3 &&  cd src && ./classifier && ./object
```
## 3. classifier result:
![图片](https://github.com/MirrorYuChen/MNN_mobilenet/blob/master/data/images/classify_result.jpg)
## 4. object result:
![图片](https://github.com/MirrorYuChen/mnn_example/blob/master/data/images/object_result.jpg)
## 5. TODO:
 - [x] add face detection, landmarker, recognizer
