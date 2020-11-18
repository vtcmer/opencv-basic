import cv2
import numpy as np
import matplotlib.pyplot as plt

## Function return cannyImage
def getCanny(img):
    # gray scale
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # reduce noise
    gaussImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
    # detection borders
    cImage = cv2.Canny(gaussImage, 50, 150)
    return cImage

def regionOfIntereset(img):
    height = img.shape[0]
    polygons = np.array([
        [(300,height) , (1200,height) , (496,180)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    maskedImage = cv2.bitwise_and(img,mask)
    return maskedImage


def displayLines(img, lns):
    lineImage = np.zeros_like(img)
    if lns is not None:
        for line in lns:
            #print(line)
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lineImage, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lineImage        

def averagedSlopeInterceptor(img, lns):
    leftFit = []
    rightFit = []
    for line in lns:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 2, full=True)
        #print(parameters)
        slope = parameters[0][0]
        intercept = parameters[0][1]
        if slope < 0:
            leftFit.append((slope, intercept))
        else:
            rightFit.append((slope, intercept))
    leftFitAverage = np.average(leftFit, axis=0)
    rightFitAverage = np.average(rightFit, axis=0)
    leftLine = makeCoordinates(img, leftFitAverage)
    rightLine = makeCoordinates(img, rightFitAverage)
    #print(leftFitAverage, 'left')
    #print(rightFitAverage, 'right')
    return np.array([leftLine, rightLine])


def makeCoordinates(img, lns):
    slope, interceptor = lns
    #print(img.shape)
    y1 = img.shape[0]
    y2 = int(y1*(3/5))
    x1 = int ((y1 - interceptor)/slope) 
    x2 = int ((y2 - interceptor)/slope)
    return np.array([x1, y1, x2, y2])

######  START MAIN ###############################################3

image = cv2.imread('img/road4.jpeg')
laneImage = np.copy(image)
cannyImage = getCanny(laneImage)
croppedImage = regionOfIntereset(cannyImage)
lines = cv2.HoughLinesP(croppedImage, 1, np.pi/180, 100, cv2.THRESH_BINARY
                        , minLineLength=40, maxLineGap=10)
averagedLines = averagedSlopeInterceptor(laneImage, lines)
linesImage = displayLines(laneImage, lines)
comboImages = cv2.addWeighted(laneImage, 0.8, linesImage, 1, 1)

cv2.imshow('result', comboImages)

cv2.waitKey(0)
#plt.imshow(cannyImage)
#plt.show()
