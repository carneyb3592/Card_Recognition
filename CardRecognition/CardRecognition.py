import cv2
from PIL import Image, ImageOps
import numpy as np
import os
os.chdir(r'C:\Users\djntr\source\repos\CardRecognition\CardRecognition')
cap = cv2.VideoCapture(0)
cap.set(3,1280)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

CARD_RATIO = 0.63
CARD_AREA_MIN = 70000
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
        self.rankImg = []
        self.rank = ""
        self.suitImg = []
        self.suit = ""
        self.valuesOnCard = []
        self.correctedImg = []

class Suits:
    def __init__(self):
        self.width = 0

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
            #image = cv2.putText(frame, 'Card', (x + int(w/2),y + int(h/2)), font, 
            #                  fontScale, color, thickness, cv2.LINE_AA)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    recognize_cards(possible_cards, frame,contours,hierarchy)
    

def recognize_cards(cards, frame,contours,hierarchy):
    for card in cards:
        card.correctedImg = makeCardReadable(card, frame)
        cv2.imshow("cropped", card.correctedImg)
        card.rankImg = cv2.resize(card.correctedImg[10:50,10:30],(40,80))
        cv2.imshow("ranking", card.rankImg)
        card.suitImg = cv2.resize(card.correctedImg[50:80,10:30],(60,90))
        cv2.imshow("suit", card.suitImg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("King.jpg",card.rankImg)


      
     
def makeCardReadable(card, frame):

    "Sourced from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"
    rect = np.zeros((4, 2), dtype = "float32")

    # List of coordinats of card from Top Left, Top Right, Bottom Left, Bottom Right
    pts = np.float32([(card.x,card.y),(card.x, card.y + card.height),(card.x + card.width, card.y),(card.x + card.width, card.y + card.height)])



    if(card.width < card.height):
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        difference = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(difference)]
        rect[3] = pts[np.argmax(difference)]
    

    if(card.width > card.height):
        s = pts.sum(axis = 1)
        rect[1] = pts[np.argmin(s)]
        rect[3] = pts[np.argmax(s)]

        difference = np.diff(pts, axis = 1)
        rect[2] = pts[np.argmin(difference)]
        rect[0] = pts[np.argmax(difference)]

    (tl, tr, br, bl) = rect
    
   
    maxWidth = 200
    
    maxHeight = 300

    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
    gray_warped =  cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	# return the warped image
    return gray_warped



while True:
    ret, frame = cap.read()
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Gray',grayImage)

    cannyN = cv2.Canny(frame, 150, 175)
    
    #cv2.imshow('Canny Normal', cannyN)
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
