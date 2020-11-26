import cv2
import numpy as np
import face_recognition

def loadImages():

    acdc = face_recognition.load_image_file("resources/faces/acdc.jpg")
    acdc = cv2.cvtColor(acdc,cv2.COLOR_BGR2RGB)

    imgTonny1 = face_recognition.load_image_file("resources/faces/tonny1.jpg")
    imgTonny1 = cv2.cvtColor(imgTonny1,cv2.COLOR_BGR2RGB)
    imgTonny2 = face_recognition.load_image_file("resources/faces/tonny2.jpg")
    imgTonny2 = cv2.cvtColor(imgTonny2,cv2.COLOR_BGR2RGB)

    imgElon1 = face_recognition.load_image_file("resources/faces/elon1.jpg")
    imgElon1 = cv2.cvtColor(imgElon1,cv2.COLOR_BGR2RGB)
    imgElon2 = face_recognition.load_image_file("resources/faces/elon2.jpg")
    imgElon2 = cv2.cvtColor(imgElon2,cv2.COLOR_BGR2RGB)
    return [acdc],[imgTonny1,imgTonny2],[imgElon1,imgElon2]

def focusFace(img):
    faceLoc = face_recognition.face_locations(img)
    faceEncode = face_recognition.face_encodings(img)
    for f in faceLoc:
        cv2.rectangle(img,(f[3],f[0]),(f[1],f[2]),(255,0,0))

def compare2Faces(face1, face2, user):
    faceLoc1 = face_recognition.face_locations(face1)[0]
    faceEncode1 = face_recognition.face_encodings(face1)[0]
    cv2.rectangle(face1,(faceLoc1[3],faceLoc1[0]),(faceLoc1[1],faceLoc1[2]),(255,0,0))
    #cv2.imshow("Face1",face1)

    faceLoc2 = face_recognition.face_locations(face2)[0]
    faceEncode2 = face_recognition.face_encodings(face2)[0]
    cv2.rectangle(face2,(faceLoc2[3],faceLoc2[0]),(faceLoc2[1],faceLoc2[2]),(255,0,0))
    
    isTheSame = face_recognition.compare_faces([faceEncode1],faceEncode2)
    distance = face_recognition.face_distance([faceEncode1],faceEncode2)
    cv2.putText(face2,f'{isTheSame} {round(distance[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    print(str(user)+str(isTheSame)+str(distance))
    cv2.imshow("Usuario::"+str(user),face2)

def checkElonvsTonny(elon, tonny):
    #Compare Elon
    compare2Faces(elon[0],elon[1],"Elon 1")
    #Compare Tonny
    compare2Faces(tonny[0],tonny[1],"Tonny 1")
    #Compare Elon
    compare2Faces(tonny[0],elon[0],"Elon 2")
    #Compare Tonny
    compare2Faces(elon[0],tonny[0],"Tonny 2")


def main():
    
    acdc, imgsTonny, imgsElon = loadImages()
    checkElonvsTonny(imgsElon,imgsTonny)
    

    '''
    faceLoc = face_recognition.face_locations(imgsTonny[0])[0]
    print(faceLoc)
    encode = face_recognition.face_encodings(imgsTonny[0])[0]
    print(encode)
    
    cont = 1
    for image in imgsElon:
        focusFace(image)
        cv2.imshow(str(cont),image)
        cont += 1
    '''
    



    cv2.waitKey(0)

main()