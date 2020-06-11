import face_recognition
import cv2
import numpy as np
import os
from os import walk

class face_model():
    # Load a sample picture and learn how to recognize it.
    def __init__(self):
        self.encodings = []
        dir_path = os.path.dirname(os.path.realpath(__file__))
        pix = os.listdir(dir_path+'/image_dir/')
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file(dir_path+'/image_dir/' + person_img)
            face_bounding_boxes = face_recognition.face_locations(face)
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                self.encodings.append(face_enc)
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        for (dirpath, dirnames, filenames) in walk(dir_path+'/image_dir/'):
            for file in filenames:
                file = os.path.splitext(file)[0]
                self.face_names.append(file)
        self.face_rect = []
        self.process_this_frame = True

    def __call__(self, body, frame):
        frame, head_rect = self.face_bbox(body, frame)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_model uses)
        frame = frame[:, :, ::-1]
        # Only process every other frame of video to save time
        if self.process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            self.face_locations = face_recognition.face_locations(frame)
            self.face_encodings = face_recognition.face_encodings(frame, self.face_locations)
            best_match_index = 0
            name = 'Unknown Face'
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(self.encodings, face_encoding, tolerance=0.5)
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[0]:
                    name = self.face_names[best_match_index]
            return name, head_rect

    @staticmethod
    def face_bbox(body, frame):
        head_points = [0, 15, 16, 17, 18]
        head = np.zeros([5, 3], dtype='float32')
        for k in range(5):
            head[k] = body[0][head_points[k]]
        if np.sum(head[:, 2]) > 5 * 0.5:
            head_rect = cv2.boundingRect(np.array(head[:, :2]))
            x, y, w, h = head_rect
            frame = frame[y - (int)(y / 1.5): y + h + (int)(y / 1.5), x - (int)(x / 4): x + w + (int)(x / 4)]
            return frame, head_rect
        else:
            head_rect = ()
            return frame, head_rect


    def recognize(self,frame):
        self.face_rect.clear()
        rgb_frame = frame[:, :, ::-1]
        # Find all the faces and face enqcodings in the frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                self.face_names.append(best_match_index)
            else:
                self.face_names.append(-1)
            self.face_rect.append((top, right, bottom, left))
        return self.face_names, self.face_rect

