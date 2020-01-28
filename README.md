# MNN_mobilenet
alibaba MNN, mobilenet classifier
# **How to use?**
## 1. convert model
'''
>> ./MNNConvert -f TF --modelFile mobilenet_v1_1.0_224_frozen.pb --MNNModel mobilenet.mnn --bizCode MNN
'''
## 2. build
'''
>> mkdir build && cd build && make -j3
>> ./classifier
'''
## 3. result
