import cv2
import numpy as np


def empty(a):
    pass

def detectColorPicker():

    cv2.namedWindow("HSV")
    cv2.resizeWindow("HSV",640,240)
    cv2.createTrackbar("Hue Min","HSV",0,179,empty)
    cv2.createTrackbar("Sat Min","HSV",59,255,empty)
    cv2.createTrackbar("Val Min","HSV",66,255,empty)
    
    cv2.createTrackbar("Hue Max","HSV",179,179,empty)
    cv2.createTrackbar("Sat Max","HSV",255,255,empty)
    cv2.createTrackbar("Val Max","HSV",255,255,empty)
    
    cap = cv2.VideoCapture(0)
    cap.set(3,640) #Height
    cap.set(4,480)  #width
    cap.set(10,100) #Brillo    

        
    while True:
        _, img = cap.read()

        if img is None:   # Verificación si terminan los frames
         break

        imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hMin = cv2.getTrackbarPos("Hue Min","HSV")
        hMax = cv2.getTrackbarPos("Hue Max","HSV")
        sMin = cv2.getTrackbarPos("Sat Min","HSV")
        sMax = cv2.getTrackbarPos("Sat Max","HSV")
        vMin = cv2.getTrackbarPos("Val Min","HSV")
        vMax = cv2.getTrackbarPos("Val Max","HSV")

        print(hMin,sMin,vMin,hMax,sMax, vMax)
        lower = np.array([hMin,sMin,vMin])
        upper = np.array([hMax,sMax, vMax])
        mask = cv2.inRange(imgHsv,lower, upper)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
     

        cv2.imshow("Output Video",img)
        cv2.imshow("Output Video Result",mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()  

#detectColorPicker()


###########################################################################################


def findColor(img, myColors,imgContour,myColorValues):
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cont = 0
    points = []
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHsv,lower, upper)
        x,y = getContours(mask,imgContour)
        #cv2.circle(imgContour,(x,y),10,(255,0,0),cv2.FILLED)
        cv2.circle(imgContour,(x,y),10,myColorValues[cont],cv2.FILLED)
        if x != 0 and y != 0:
            points.append([x,y,cont])
        cont +=1
        #cv2.imshow(str(color[0]),mask)
    return points

def getContours(img,imgContour):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            #cv2.drawContours(imgContour,cnt,-1,(255,0,0),3)
            peri = cv2.arcLength(cnt,True)
            aprox = cv2.approxPolyDP(cnt,0.02*peri,True)
            x, y, w, h = cv2.boundingRect(aprox)

    return x+w//2,y

def drawOnCavas(myPoints, myColorValues,imgContour):

    for point in myPoints:
        cv2.circle(imgContour,(point[0],point[1]),10,myColorValues[point[2]],cv2.FILLED)




def main():

    myColors = [[16,96,255,179,255,255],[86,42,201,98,255,255]]
    #,[86,0,255,99,144,255

    myColorValues = [[51,206,255],[51,255,60]]   #BGR

    myPoints =  []            #[x,y, colorId]

    cap = cv2.VideoCapture(0)
    cap.set(3,640) #Height
    cap.set(4,480)  #width
    cap.set(10,100) #Brillo    

        
    while True:
        success, frame = cap.read()

        if frame is None:   # Verificación si terminan los frames
         break

        imgContour = frame.copy()
        newPoints = findColor(frame,myColors,imgContour,myColorValues)
        if (len(newPoints) != 0):
            for p in newPoints:
                myPoints.append(p)
        
        if len(myPoints) != 0:
            drawOnCavas(myPoints,myColorValues,imgContour)
 
        
        cv2.imshow("Output Video",imgContour)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()  




main()