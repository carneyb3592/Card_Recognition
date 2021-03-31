import cv2
from PIL import Image, ImageOps
import numpy as np
cap = cv2.VideoCapture(0)
cap.set(3,1280)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

CARD_RATIO = 0.63
CARD_AREA_MIN = 25000
CARD_AREA_MAX = 100000
SUIT_AREA_MIN = 200
SUIT_AREA_MAX = 5000
class Card:
    def __init__(self):
        self.boundingRect = []
        self.width = 0
        self.height = 0
        self.x = 0
        self.y = 0
        self.rank = ""
        self.suit = ""


def detect_Cards(contours, hierarchy, frame):
    possible_cards = []
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        area = w * h
        areaActual = cv2.contourArea(contour)
        extent = float(areaActual)/area
        #ratio_of_width_to_height = w/h
        if area < CARD_AREA_MAX and area > CARD_AREA_MIN  and extent > 0.7:
            card = Card()
            card.boundingRect = [x,y,w,h]
            card.width = w
            card.height = h
            card.x = x
            card.y = y
            possible_cards.append(card)
            font = cv2.FONT_HERSHEY_SIMPLEX
  
            # fontScale
            fontScale = 1
   
            # Blue color in BGR
            color = (255, 0, 0)
  
            # Line thickness of 2 px
            thickness = 2
   
            # Using cv2.putText() method
            image = cv2.putText(frame, 'Card', (x + int(w/2),y + int(h/2)), font, 
                               fontScale, color, thickness, cv2.LINE_AA)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    recognize_cards(possible_cards,contours, hierarchy, frame)
    

def recognize_cards(cards,contours, hierarchy, frame):
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        area = w * h
        if area < SUIT_AREA_MAX and area > SUIT_AREA_MIN:
            for card in cards:
                if(x > card.x and x < card.x+card.width and y > card.y and y < card.y+card.height):
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)


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
