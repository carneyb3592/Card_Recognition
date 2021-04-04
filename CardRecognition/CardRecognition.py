import cv2
from PIL import Image, ImageOps
import numpy as np
import os
os.chdir(r'C:\Users\djntr\source\repos\CardRecognition\CardRecognition')
cap = cv2.VideoCapture(0)
cap.set(3,1280)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError('Cannot open webcam')

CARD_RATIO = 0.63
CARD_AREA_MIN = 25000
CARD_AREA_MAX = 80000
SUIT_AREA_MIN = 200
SUIT_AREA_MAX = 5000
FILEPATH = r'C:\Users\djntr\source\repos\CardRecognition\CardRecognition\CardImages\\'


class Card:
    def __init__(self):
        self.corners = []
        self.contour = []
        self.boundingRect = []
        self.width = 0
        self.height = 0
        self.x = 0
        self.y = 0
        self.rankImg = []
        self.rank = ''
        self.value = 0
        self.suitImg = []
        self.suit = ''
        self.valuesOnCard = []
        self.correctedImg = []

class Suit:
    def __init__(self):
       self.image = []
       self.name = ''

class Value:
    def __init__(self):
       
       self.image = []
       self.name = ''
       self.value = 0

def process_suits():
    suit_list = []
    counter = 0

    for Suits in ['Clubs','Diamonds','Spades','Hearts']:
        suit_list.append(Suit())
        suit_list[counter].name = Suits
        suit_list[counter].image = cv2.resize(cv2.imread((FILEPATH + Suits + '.jpg'),cv2.COLOR_BGR2GRAY),(60,90))
        counter = counter + 1
    return suit_list
def process_values():

    value_list = []
    counter = 0

    for Values in ['Ace','Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten','Jack','Queen','King']:
        value_list.append(Value())
        value_list[counter].name = Values
        value_list[counter].image = cv2.resize(cv2.imread((FILEPATH + Values + '.jpg'),cv2.COLOR_BGR2GRAY),(40,80))
        value_list[counter].value = counter + 1
        counter = counter + 1
    return value_list

def detect_Cards(contours, frame):
    possible_cards = []
    for contour in contours: 
        insideCard = False

        (x,y,w,h) = cv2.boundingRect(contour)
        #area = w * h

        areaActual = cv2.contourArea(contour)
        # Finds number of sides of the object
        sides = cv2.approxPolyDP(contour, 0.1* cv2.arcLength(contour,True) ,True)
        #extent = float(areaActual)/area
        #ratio_of_width_to_height = w/h
        if areaActual < CARD_AREA_MAX and areaActual > CARD_AREA_MIN  and len(sides) == 4:
            
            #Checks for cases where face cards have boxes inside that would normally come up as a card
            if len(possible_cards) != 0:
                for card in possible_cards:
                    if x > card.x and x < card.x+card.width and y > card.y and y < card.y + card.height:
                        insideCard = True
            if not insideCard:
                
                corners = np.float32(sides)
                card = Card()
                card.contour = contour
                card.boundingRect = [x,y,w,h]
                card.width = w
                card.height = h
                card.x = x
                card.y = y
                card.corners = corners
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
                #cv2.drawContours(frame, contour, -1, (0,255,0), 2)
                #print(card.contour)
    recognize_cards(possible_cards, frame,contours,hierarchy)
    

def recognize_cards(cards, frame,contours,hierarchy):
    suits = process_suits()
    values = process_values()

    for card in cards:
        card.correctedImg = makeCardReadable(card, frame)
        cv2.imshow('Rcards',card.correctedImg)
        
        card.suitImg, card.rankImg = isolateSuitsValues(card.correctedImg,frame)
        cv2.imshow('Rank',card.rankImg)
        cv2.imshow('Suit',card.suitImg)
        #c = cv2.waitKey(1)
        #if c == ord('q'):
         #   cv2.imwrite("Four.jpg",card.rankImg)
          #  cv2.imwrite("Clubs.jpg",card.suitImg)

        #card.suit = detect_suit(card,suits)
        #card.rank, card.value = detect_value(card,values)
        #print(card.suit,card.rank)

def detect_suit(card,suits):
    best_match = 10000
    best_name = 'Undetermined'

    for suit in suits:
        
        diff = cv2.absdiff(card.suitImg,suit.image)
      
        diff_value = int(np.sum(diff)/255)

        if diff_value < best_match:
            best_match = diff_value
            best_name = suit.name
    return best_name

def detect_value(card,values):
    best_match = 10000
    best_name = 'Undetermined'
    best_value = 0
    for value in values:
        
        diff = cv2.absdiff(card.rankImg,value.image)

        diff_value = int(np.sum(diff)/255)

        if diff_value < best_match:
            best_match = diff_value
            best_name = value.name
            best_value = value.value
    return best_name, best_value

def makeCardReadable(card, frame):

    'Sourced from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/'
    rect = np.zeros((4, 2), dtype = 'float32')
    s = card.corners.sum(axis = 2)

    if(card.width < card.height):
        rect[0] = card.corners[np.argmin(s)]
        rect[2] = card.corners[np.argmax(s)]
        difference = np.diff(card.corners, axis = 2)
        rect[1] = card.corners[np.argmin(difference)]
        rect[3] = card.corners[np.argmax(difference)]
    

    if(card.width > card.height):
        rect[1] = card.corners[np.argmin(s)]
        rect[3] = card.corners[np.argmax(s)]

        difference = np.diff(card.corners, axis = 2)
        rect[2] = card.corners[np.argmin(difference)]
        rect[0] = card.corners[np.argmax(difference)]

    (tl, tr, br, bl) = rect
    
   
    maxWidth = 200
    
    maxHeight = 300

    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = 'float32')
	# compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
    gray_warped =  cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	# return the warped image
    return gray_warped

def isolateSuitsValues(cardImg,frame):
    suit_img = []
    value_img = []
   
    rankImg = cv2.resize(cardImg[0:45,0:32],(40,80))
    cannyR = cv2.Canny(rankImg, 150, 175)
    retR, threshR = cv2.threshold(cannyR, 127, 255, 0)
    contoursR, hierarchyR = cv2.findContours(threshR,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rankImg,contoursR, -1, (0,255,0), 0)
    for contour in contoursR:
         (x,y,w,h) = cv2.boundingRect(contour)
         cv2.rectangle(rankImg, (x,y), (x+w,y+h), (0,255,0), 2)
    suitImg = cv2.resize(cardImg[40:84,0:32],(60,90))
    cannyS = cv2.Canny(suitImg, 150, 175)
    retS, threshS = cv2.threshold(cannyS, 127, 255, 0)
    contoursS, hierarchyS = cv2.findContours(threshS,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(suitImg,contoursS, -1, (0,255,0), 0)
    for contour in contoursS:
         (x,y,w,h) = cv2.boundingRect(contour)
         cv2.rectangle(suitImg, (x,y), (x+w,y+h), (0,255,0), 2)
    
    #for contour in contours:
    #   (x,y,w,h) = cv2.boundingRect(contour)
     #  cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    return suitImg, rankImg

while True:
    cv2.waitKey(100)
    ret, frame = cap.read()
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA

    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Gray',grayImage)

    cannyN = cv2.Canny(frame, 150, 175)
    
    #cv2.imshow('Canny Normal', cannyN)
    ret, thresh = cv2.threshold(cannyN, 127, 255, 0)
   
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    detect_Cards(contours, frame)
    
    #print(hierarchy)    
    #cv2.drawContours(frame, contours, -1, (0,255,0), 0)
    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    #print('Number of Contours found = ' + str(len(contours)))
    if c == 27:
        
        break




cap.release()
cv2.destroyAllWindows()
