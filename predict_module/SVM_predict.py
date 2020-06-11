import numpy as np
import cv2
import os
import joblib

class SVM_predict():
    def __init__(self):
        self.SVM = self.loadSVMmodel()
        self.gesture = ['Ok', 'One', 'Two', 'Three']

    def loadSVMmodel(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        SVM = joblib.load(dir_path+"\SVM_model.m")
        return SVM

    def __call__(self, hand):
        result = self.SVM.predict(hand)
        return self.gesture[result[0]]

    @staticmethod
    def hand_box(hand):
        if np.sum(hand[:, 2]) > 21 * 0.5:
            rect = cv2.boundingRect(np.array(hand[:, :2]))
            return rect
        else:
            return ()