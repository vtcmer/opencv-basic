import cv2
import numpy as np



#resize Image
def showFace():

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_alt2.xml')

    img = cv2.imread("resources/stark.jpg")
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(imgGray,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
   
    cv2.imshow("Output",img)
    cv2.waitKey(0)

showFace()