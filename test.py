import time

import torch

from face_detection import RetinaFace

if __name__ == "__main__":
    detector = RetinaFace(0)
    img = torch.zeros(35, 2500, 2500, 3).to(torch.uint8) * 255
    # size = 500
    # batch_input = [img] * size
    start = time.time()
    faces = detector(img)
    # import cv2
    # t = faces[0][0]
    # for img_faces, img2 in zip(faces, batch_input):
    #     box, landmarks, score = img_faces[0]
    #     box = box.astype(np.int)
    #     cv2.imshow("", img2[box[1]:box[3], box[0]:box[2]])
    #     cv2.waitKey(0)
