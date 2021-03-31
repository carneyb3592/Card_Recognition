import cv2
from PIL import Image, ImageOps
import numpy as np
cap = cv2.VideoCapture(0)
cap.set(3,1280)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

CARD_AREA_MIN = 25000
CARD_AREA_MAX = 100000

def detect_Cards(contours, hierarchy, frame):
    possible_cards = []
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        area = w * h
        areaActual = cv2.contourArea(contour)
        extent = float(areaActual)/area
        if area < CARD_AREA_MAX and area > CARD_AREA_MIN and w < h and extent > 0.3:
            possible_cards.append([x,y,w,h])
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    recognize_cards(possible_cards)
    

def recognize_cards(cards):
    for count in cards:
        print("Contour!")

while True:
    ret, frame = cap.read()
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)




    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Gray',grayImage)

    cannyN = cv2.Canny(frame, 150, 175)
    cv2.imshow('Canny Normal', cannyN)
    ret, thresh = cv2.threshold(cannyN, 127, 255, 0)
   
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detect_Cards(contours, hierarchy,frame)
    #print(hierarchy)    
    #cv2.drawContours(frame, contours, -1, (0,255,0), 0)
    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    #print("Number of Contours found = " + str(len(contours)))
    if c == 27:
        
        break




cap.release()
cv2.destroyAllWindows()
