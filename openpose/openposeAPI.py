import os
import sys
from sys import platform

class openposeAPI():
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.dirname(dir_path)
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/openpose');
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/openpose;' + dir_path + '/3rdparty;'
            import pyopenpose as op

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = dir_path + "/models/"
        params["net_resolution"] = "80x80"
        params["hand"] = True
        #params["disable_blending"] = True
        # Construct it from system arguments
        # Fixed the handRectangles to only 1 person and 1 big rectangle, don't have to keep changing rectangle

        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()
        self.datum = op.Datum()

    def get_skeleton(self, frame):
        self.datum.cvInputData = frame
        self.opWrapper.emplaceAndPop([self.datum])
        frame = self.datum.cvOutputData
        body = self.datum.poseKeypoints  # get body coordinates
        hand = self.datum.handKeypoints  # get hand coordinates
        face = self.datum.faceKeypoints  # get face coordinates
        return frame, (body, hand, face)
