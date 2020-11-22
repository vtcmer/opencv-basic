import cv2
import numpy as np

#https://www.youtube.com/watch?v=WQeoO7MI0Bs

# Show image
def showImage():
    img = cv2.imread("resources/stark.jpg")
    cv2.imshow("Output",img)
    cv2.waitKey(0)


#showImage()


def showImageColor():
    img = cv2.imread("resources/stark.jpg")
    kernel = np.ones((5,5),np.uint8)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # escala de grises
    blur = cv2.GaussianBlur(gray,(7,7),0) # Quitar ruido
    canny = cv2.Canny(img,100,100) #  Resaltar bordes
    dialatation = cv2.dilate(canny,kernel, iterations=1)
    erode = cv2.erode(dialatation,kernel,iterations=1)

    

    cv2.imshow("Output",img)
    cv2.imshow("Output Gray",gray)
    cv2.imshow("Output Blur",blur)
    cv2.imshow("Output Canny",canny)
    cv2.imshow("Output dialatation",dialatation)
    cv2.imshow("Output erode",erode)
    cv2.waitKey(0)

#showImageColor()


def showVideo():
    cap = cv2.VideoCapture("resources/road.mp4")
        
    while True:
        success, frame = cap.read()

        if frame is None:   # Verificación si terminan los frames
         break
        
        cv2.imshow("Output Video",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()         

#showVideo()

def showCamera():
    cap = cv2.VideoCapture(0)
    cap.set(3,640) #Height
    cap.set(4,480)  #width
    cap.set(10,100) #Brillo    

        
    while True:
        success, frame = cap.read()

        if frame is None:   # Verificación si terminan los frames
         break
        
        cv2.imshow("Output Video",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()      

#showCamera()  