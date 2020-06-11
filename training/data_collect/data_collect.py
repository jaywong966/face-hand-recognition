import cv2
import csv
import numpy as np
from openpose.openposeAPI import openposeAPI
from predict_module.data_preprocessing import data_preprocessing

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    f = open('test.csv', 'w', encoding='utf-8',newline='')
    csv_writer = csv.writer(f)
    data_record = np.empty([1, 40], dtype='int64')
    openpose = openposeAPI()
    data_preprocessing = data_preprocessing()
    i = 0
    while cam.isOpened():
        ret, frame = cam.read()
        frame, skeletons = openpose.get_skeleton(frame)
        key = cv2.waitKey(delay=1)
        if key == 27:
            break
        elif key == ord('s'):
            vector = data_preprocessing(skeletons[1][1][0], skeletons[1][0][0])
            data_record = np.delete(data_record, 0, axis=0)
            x = vector[0].flatten()
            data_record = np.append(data_record, x[np.newaxis, :], axis=0)
            i += 1
            csv_writer.writerows(data_record)
            print("Recorded! " ,i)
        cv2.imshow("face-hand-recognition", frame)
    cv2.destroyAllWindows()