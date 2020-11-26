import cv2
import numpy as np

MIN_AREA = 500
COLOR = (255,0,0)

def main():

    plateCascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_russian_plate_number.xml')

    img = cv2.imread("resources/russian_plate.jpg")

    ###########
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    nPlates = plateCascade.detectMultiScale(imgGray,1.1,4)
    print(nPlates.shape)
    for (x,y,w,h) in nPlates:
        area = w*h
        print(area)
        if area > MIN_AREA:
            cv2.rectangle(img,(x,y),(x+w,y+h),COLOR,2)
            cv2.putText(img,"Number Plate ",(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,COLOR,2)

            imgRoi = img[y:y+h,x:x+w]
            cv2.imshow("ROI",imgRoi)

    cv2.imshow("Plate",img)

    if cv2.waitKey(0):
        cv2.imwrite("resources/scanner/n_plate_.jpg",imgRoi)
        cv2.rectangle(img,(0,200),(640,300),(0,255,0),cv2.FILLED)
        cv2.putText(img,"Scan save",(150,265),cv2.FONT_HERSHEY_DUPLEX,2,COLOR,2)
        cv2.imshow("Result",img)
        cv2.waitKey(500)

main()