import cv2
import numpy as np

#resize Image
def resizeImage():
    img = cv2.imread("resources/stark.jpg")
    print(img.shape) # (heigth, width, channels)

    imgResize = cv2.resize(img,(800,300))
    print(imgResize.shape) # (heigth, width, channels)

    imgCropped = img[0:200,200:500] # Take a piece of the image

    cv2.imshow("Output",img)
    cv2.imshow("Output Resize",imgResize)
    cv2.imshow("Output Cropped",imgCropped)
    cv2.waitKey(0)

#resizeImage()

def showGeometrics():
    img = np.zeros((512,512,3),np.uint8)
    #print(img.shape)
    #img[:] = 255,0,0 # img[200:300,100:300] = 255,0,0

    #cv2.line(img,(0,0),(300,300),(0,255,0),3)
    cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3)
    cv2.rectangle(img,(0,0),(250,350),(100,100,255),5)
    cv2.rectangle(img,(100,100),(150,250),(0,0,255),cv2.FILLED)
    cv2.circle(img,(400,50),30,(255,255,0),5)
    cv2.putText(img," OPENCV ",(300,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),1)

    cv2.imshow("Geometrics", img)

    cv2.waitKey(0)

#showGeometrics();

def showCardsTranslate():
    img = cv2.imread("resources/cards.jpg")

    w,h = 200,350

    pt1 = np.float32([[336,65],[461,115],[255,256],[385,315]])
    pt2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pt1,pt2)
    imgOutput = cv2.warpPerspective(img,matrix,(w,h))

    cv2.imshow("Output",img)
    cv2.imshow("Output Transform",imgOutput)
   
    cv2.waitKey(0)

#showCardsTranslate()


def empty(a):
    pass

def showColorDetection():
    
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars",640,240)
    cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
    cv2.createTrackbar("Hue Max","TrackBars",179,179,empty)
    cv2.createTrackbar("Sat Min","TrackBars",59,255,empty)
    cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
    cv2.createTrackbar("Val Min","TrackBars",66,255,empty)
    cv2.createTrackbar("Val Max","TrackBars",255,255,empty)
    

    while True:
        img = cv2.imread("resources/car.jpg")

        imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hMin = cv2.getTrackbarPos("Hue Min","TrackBars")
        hMax = cv2.getTrackbarPos("Hue Max","TrackBars")
        sMin = cv2.getTrackbarPos("Sat Min","TrackBars")
        sMax = cv2.getTrackbarPos("Sat Max","TrackBars")
        vMin = cv2.getTrackbarPos("Val Min","TrackBars")
        vMax = cv2.getTrackbarPos("Val Max","TrackBars")

        print(hMin,hMax, sMin, sMax, vMin, vMax)

        lower = np.array([hMin,sMin,vMin])
        upper = np.array([hMax,sMax, vMax])
        mask = cv2.inRange(imgHsv,lower, upper)
        imgResult = cv2.bitwise_and(img, img,mask=mask)


        cv2.imshow("Output",img)
        cv2.imshow("Output HSV",imgHsv)
        cv2.imshow("Output Mask",mask)
        cv2.imshow("Output Image Result", imgResult)
        cv2.waitKey(1)

#showColorDetection()


def getContours(img,imgContour):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
       
        if area > 500:
            #print(area)
            cv2.drawContours(imgContour,cnt,-1,(255,0,0),3)
            peri = cv2.arcLength(cnt,True)
            #print(peri)
            aprox = cv2.approxPolyDP(cnt,0.02*peri,True)
            #print(len(aprox))
            objColor = len(aprox)
            x, y, w, h = cv2.boundingRect(aprox)

            objDescription = "NaN"
            if objColor == 3:
                objDescription = 'Tri'
            elif  objColor == 4:
                aspRatio = w/float(h)
                if (aspRatio > 0.95 and aspRatio < 1.05):
                    objDescription = "Cua"
                else:
                    objDescription = "Rec"
            elif  objColor == 5:
                objDescription = 'Pen'
            elif  objColor == 6:
                objDescription = 'Hex'
            else:
                aspRatio = w/float(h)
                print(aspRatio)
                if (aspRatio == 1):
                    objDescription = "Octo"
                else:
                    objDescription = "Cir"

            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(imgContour,objDescription,(x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),2)

def showShapes():
    img = cv2.imread("resources/shapes.jpeg")

    imgContour = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(img,(7,7),1)
    imgCanny = cv2.Canny(imgBlur,50,50)
    
    getContours(imgCanny,imgContour)

    cv2.imshow("Output",img)
    cv2.imshow("Output Gray",imgGray)
    cv2.imshow("Output Blur",imgBlur)
    cv2.imshow("Output Canny",imgCanny)
    cv2.imshow("Output Contours",imgContour)
    cv2.waitKey(0)

#showShapes()