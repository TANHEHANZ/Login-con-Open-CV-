import cv2
from mtcnn import MTCNN

img = cv2.cvtColor(cv2.imread("imagen_con_mascarilla.jpg"), cv2.COLOR_BGR2RGB)
detector = MTCNN()
detector.detect_faces(img)
