import numpy as np
import cv2
import os
from face_model.face_recognition import face_model
from openpose.openposeAPI import openposeAPI
from predict_module.data_preprocessing import data_preprocessing
from predict_module.SVM_predict import SVM_predict



class main():
    def __init__(self):
        super().__init__()
        self.face_model = face_model()
        self.openposeAPI = openposeAPI()
        self.data_preprocessing = data_preprocessing()
        self.SVM_predict = SVM_predict()
        self.person_num = 0
        self.user_message = []

    def update_frame(self,frame):
        self.user_message.clear()      # clear
        frame, skeletons = self.openposeAPI.get_skeleton(frame)
        # get frame and skeletons's coordinate
        self.face_recognition(skeletons, frame)
        # Face recognition and put rectangele on the head
        user_message = self.gesture_recognition(skeletons, frame)
        return frame, user_message

    def face_recognition(self, skeletons, frame):
        body = skeletons[0]
        if body.size == 1:
            return ()   # return empty rect if nobody in a frame
        else:
            self.person_num = body.shape[0]
            for i in range(self.person_num):
                user_name, head_rect = self.face_model(body, frame)
                # get user id and head rect by use face module
                if head_rect:
                    x, y, w, h = head_rect
                    cv2.rectangle(frame, (x - (int)(x / 4), y - (int)(y / 1.5)), (x + w + (int)(x / 4), y + h + (int)(y / 1.5)), (255, 255, 255))
                    cv2.putText(frame, user_name, (x - (int)(x / 4), y - (int)(y / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
                self.user_message.append([user_name])   # record user name


    def gesture_recognition(self, skeletons, frame):
        hands = skeletons[1]
        if np.min(hands[1]) == 0:
            for i in range(self.person_num):
                self.user_message[i].append('Unkonwn Gesture')
            return self.user_message
        else:
            user_message = np.array(self.user_message)
            user_message = np.resize(user_message,(self.person_num,2))
            # Refactor this array, sometime openpose api will recognize two people but only one in the frame
            for i in range(self.person_num):
                hand_vector = self.data_preprocessing(hands[1][i], hands[0][i])
                # Turning coordinate into unit vector(right hand, left hand)
                result = self.SVM_predict(hand_vector)      # predict result (Only predict right hand)
                for hand in hands:
                    hand_rect = self.SVM_predict.hand_bbox(hand[i])
                    user_message[i][1] = result     # record gesture result
                    if hand_rect:
                        x, y, w, h = hand_rect
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255))
                        cv2.putText(frame, result, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            user_message = user_message.tolist()
            return user_message



if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    main = main()
    while (cv2.waitKey(1) != 27):
        ret, frame = cam.read()
        frame, user_message = main.update_frame(frame)
        print(user_message)
        cv2.imshow("face-hand-recognition", frame)
    cv2.destroyAllWindows()