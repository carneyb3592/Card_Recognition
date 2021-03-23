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




    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Gray',grayImage)

    cannyN = cv2.Canny(frame, 150, 175)
    cv2.imshow('Canny Normal', cannyN)
    ret, thresh = cv2.threshold(cannyN, 127, 255, 0)
   
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(hierarchy)    
    cv2.drawContours(frame, contours, -1, (0,255,0), 0)
    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    #print("Number of Contours found = " + str(len(contours)))
    if c == 27:
        
        break

cap.release()
cv2.destroyAllWindows()
