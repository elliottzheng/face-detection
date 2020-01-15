import time

from skimage import io

from face_detection import RetinaFace

if __name__ == "__main__":
    detector = RetinaFace()
    img = io.imread('examples/obama.jpg')
    size = 1
    batch_input = [img] * size
    start = time.time()
    faces = detector(batch_input)
    print(time.time() - start)
    start = time.time()
    for i in range(size):
        faces = detector(img)
    print(time.time() - start)