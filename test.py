import time
import torch
from face_detection import RetinaFace
import numpy as np
import cv2

if __name__ == "__main__":
    detector=RetinaFace(0,model_path="face_detection\weights\Resnet50_Final.pth",network="resnet50")
    # detector=RetinaFace(0)
    img=cv2.cvtColor(cv2.imread("examples/obama.jpg"),cv2.COLOR_BGR2RGB)
    for i in range(5):
        start = time.time()
        faces = detector(img)
        print(time.time()-start)
        box, landmarks, score = faces[0]
        box = box.astype(np.int)
        cv2.imshow("", img[box[1]:box[3], box[0]:box[2]])
        cv2.waitKey(0)
