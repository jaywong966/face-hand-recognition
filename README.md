# Face-Hand-Recognition
## Introduction
Recognize user gestures and faces based on face_recognition and openpose
![image](https://github.com/jaywong966/face-hand-recognition/blob/master/doc/gesture.gif)

## Dependence
+ sklearn
+ numpy
+ openpose
+ face_recognition
+ opencv
+ Python 3.7

## Installation
``` git clone <repositories> ```
### Openpose
Unzip 3rdparty and model to project root path. This ‘3rdparty’ folder contains openpose python API Dependent file. **If you are not running under win10 and Nvdia platform, you must recompile openpose again**  
```https://drive.google.com/file/d/1vfRpQ3feUDNvEzO1CtgvnvSKcgmyvphy/view?usp=sharing``` ---3rdparty and models
### Face_recogntion
Mac & Linux:  
``` pip3 install face_recognition ```  
Windows:  
[@masoudr's Windows 10 installation guide (dlib + face_recognition)](https://github.com/ageitgey/face_recognition/issues/175#issue-257710508). 
## Output
``` user_message : [[user name, gesture]]```
## Train your own gesture

You can collect your hand data by ``` data_collect.py ``` and train your SVM by ``` SVM_train.py ```

## References
[Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)  
[Face_recognition](https://github.com/ageitgey/face_recognition)
