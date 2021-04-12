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
SUIT_AREA_MIN = 50
SUIT_AREA_MAX = 150
RANK_AREA_MIN = 50
RANK_AREA_MAX = 150
FILEPATH = r'C:\Users\djntr\source\repos\CardRecognition\CardRecognition\CardImages\\'

final_cards = []

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
        self.suitDict = {}
        self.valueDict = {}
        self.thresh = []

class Card_image:
     def __init__(self):
       
       self.image1 = []
       self.image2 = []
       self.name = ''
       self.value = 0
       self.suit = ""
       self.rank = ""

def process_cards():
    card_list = []
    overallCounter = 0
    for Suits in ['Clubs','Diamonds','Spades','Hearts']:
        counter = 1
        for Values in ['Ace','Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten','Jack','Queen','King']:
            card_list.append(Card_image())
            card_list[overallCounter].name = Values + ' of ' + Suits
            card_list[overallCounter].image1 = cv2.imread((FILEPATH + Suits + Values + '.jpg'),cv2.cv2.COLOR_BGR2GRAY)
            card_list[overallCounter].image2 = cv2.flip(card_list[overallCounter].image1,-1)
            card_list[overallCounter].value = counter
            card_list[overallCounter].rank = Values
            card_list[overallCounter].suit = Suits
            counter += 1
            overallCounter += 1
    return card_list



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
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                #cv2.drawContours(frame, contour, -1, (0,255,0), 2)
                #print(card.contour)
    possible_cards = recognize_cards(possible_cards, frame,contours,hierarchy)
    return possible_cards
    

def recognize_cards(cards, frame,contours,hierarchy):
    card_images = process_cards()
    #print(card_images)
    for card in cards:
        card.correctedImg = makeCardReadable(card, frame)
        cv2.imshow('Rcards',card.correctedImg)
        
        cannyR = cv2.Canny(card.correctedImg, 150, 175)
        ret, thresh = cv2.threshold(cannyR, 127, 255, 0)
        thresh = cv2.adaptiveThreshold(cannyR,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
        #contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        card.thresh = thresh
        cv2.imshow('cardd',thresh)
        
        
        
        ##Code to make and save images
        #cv2.imwrite("DiamondsAce.jpg",thresh)
        
        card.suit,card.rank = determineBestMatch(card,card_images)
        drawCards(card,frame)
        
        #print(card.suit,card.rank)
    return cards

def drawCards(card,frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    card_text = card.rank + ' of ' + card.suit
    # fontScale
    fontScale = 1
   
    # Blue color in BGR
    color = (255, 0, 0)
  
    # Line thickness of 2 px
    thickness = 2
   
    # Using cv2.putText() method
    cv2.putText(frame, card_text, (card.x,card.y + int(card.height/2)), font, fontScale, color, thickness, cv2.LINE_AA)

def determineBestMatch(card,card_images):
    best_match = 10000
    best_name = 'Undetermined'
    best_suit = 'Undetermined'
    best_rank = 'Undetermined'
    best_value = 0
    for card_image in card_images:
        
        img1 = cv2.GaussianBlur(card.thresh,(5,5),5)
        img2 = cv2.GaussianBlur(card_image.image1,(5,5),5)
        img3 = cv2.GaussianBlur(card_image.image2,(5,5),5)
        
        diff1 = cv2.absdiff(img1,img2)
        diff1  = cv2.GaussianBlur(diff1,(5,5),5)
        diff_value1 = int(np.sum(diff1)/255)
        
        diff2 = cv2.absdiff(img1,img3)
        diff2 = cv2.GaussianBlur(diff2,(5,5),5) 

        diff_value2 = int(np.sum(diff2)/255)

        if diff_value1 < best_match or diff_value2 < best_match:
            if diff_value1 < diff_value2:
                best_match = diff_value1
            else:
                best_match = diff_value2
            best_name = card_image.name
            best_suit = card_image.suit
            best_rank = card_image.rank
            best_value = card_image.value
    return best_suit, best_rank


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
    gray_warped = cv2.resize(gray_warped,(200,300))
	# return the warped image
    return gray_warped




    

while True:
    cv2.waitKey(200)
    ret, frame = cap.read()
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA

    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Gray',grayImage)

    cannyN = cv2.Canny(frame, 150, 175)
    
    #cv2.imshow('Canny Normal', cannyN)
    ret, thresh = cv2.threshold(cannyN, 127, 255, 0)
   
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    final_cards = detect_Cards(contours, frame)
    #print(cards)
    #print(hierarchy)    
    #cv2.drawContours(frame, contours, -1, (0,255,0), 0)
    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    #print('Number of Contours found = ' + str(len(contours)))
    if c == 27:
        
        break




cap.release()
cv2.destroyAllWindows()
