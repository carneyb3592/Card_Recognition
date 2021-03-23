import cv2
from PIL import Image, ImageOps
import numpy as np
cap = cv2.VideoCapture(0)
cap.set(3,1280)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)

    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Gray',grayImage)

    cannyN = cv2.Canny(frame, 150, 175)
    cv2.imshow('Canny Normal', cannyN)
    c = cv2.waitKey(1)
    if c == 27:
        
        break

cap.release()
cv2.destroyAllWindows()
