import cv2
from PIL import Image, ImageOps
cap = cv2.VideoCapture(0)
cap.set(3,1280)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    if c == 27:
        #Writes a grascaled image right before shutdown
        cv2.imwrite("firstimage.jpg",frame)
        nImage = Image.open("firstimage.jpg")
        n2Image = ImageOps.grayscale(nImage)
        n2Image.save("Grayscaled.jpg")
        break

cap.release()
cv2.destroyAllWindows()
