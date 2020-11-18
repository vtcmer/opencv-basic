import cv2
import numpy as np
import matplotlib.pyplot as plt
#https://github.com/galenballew/SDC-Lane-and-Vehicle-Detection-Tracking/blob/master/Part%20I%20-%20Simple%20Lane%20Detection/P1.ipynb
## Recupera la escala de grises
def getGrayScale(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayImage

def getImageHSV(img):
    hsv = cv2.cvtColor(image, cv2.COLOR_HLS2BGR)
    return hsv

def applyGrayHsvMask(grayImage, imgHsv):
    lower_yellow = np.array([20, 100, 100], dtype = "uint8")
    upper_yellow = np.array([30, 255, 255], dtype = "uint8")
    mask_yellow = cv2.inRange(imgHsv, lower_yellow, upper_yellow)

    mask_white = cv2.inRange(grayImage, 200, 255)
    
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(edges, mask_yw)
    return mask_yw_image

## Function return cannyImage
def getCanny(img):
    gaussImage = cv2.GaussianBlur(img, (5, 5), 0)
    # detection borders
    cImage = cv2.Canny(gaussImage, 50, 150)
    return cImage



def displayLines(img, lns):
    lineImage = np.zeros_like(img)
    if lns is not None:
        for line in lns:
            x1, y1, x2, y2 = line[0]
            cv2.line(lineImage, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lineImage 

def regionOfIntertes(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


######  START MAIN ###############################################3

baseImage = cv2.imread('img/road5.jpeg')
image = np.copy(baseImage)

imshape = image.shape
lower_left = [imshape[1]/9,imshape[0]]
lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
top_left = [imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/10]
top_right = [imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/10]
vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
print(lower_left)
print(lower_right)
print(top_left)
print(top_right)
print(vertices)


img_hsv = getImageHSV(image)
edges = getGrayScale(image)
mask_yw_image = applyGrayHsvMask(edges, img_hsv)

cannyImage = getCanny(mask_yw_image)

roiImage = regionOfIntertes(cannyImage,vertices)

lines = cv2.HoughLinesP(roiImage, 3, np.pi/180, 30, np.array([]), minLineLength=100, maxLineGap=150)
#lines = cv2.HoughLinesP(roiImage, 4, np.pi/180, 30, minLineLength=100, maxLineGap=180)
linesImage = displayLines(image, lines)
finishImage = cv2.addWeighted(image, 0.8, linesImage, 1, 1)
cv2.imshow('Road', finishImage)


cv2.waitKey(0)
#plt.imshow(cannyImage)
#plt.show()
