import cv2
import numpy as np

from face_detection import RetinaFace

if __name__ == "__main__":
    detector = RetinaFace(0)
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector(img)
        box, landmarks, score = faces[0]
        box = box.astype(np.int)
        cv2.rectangle(
            img, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=2
        )
        cv2.imshow("", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
