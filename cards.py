
import cv2
from PIL import Image, ImageOps
import numpy as np

suits = ["Clubs","Diamonds","Hearts","Spades"]
values = ["Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten","Jack","Queen","King","Ace"]
def makeDeck(suits,values):
    deck = []
    for suit in suits:
        for value in values:
            deck.append([suit,value])

    return deck

print(makeDeck(suits,values))