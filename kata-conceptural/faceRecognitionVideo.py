import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#Devuelve el listado de nombre de ficheros y la url
def readImagesFromDisk():
    path="resources/faces/control"
    files = os.listdir(path)
    images = []
    fileNames = []
    print(files)
    for file in files:
        urlImage = cv2.imread(f'{path}/{file}')
        images.append(urlImage)
        fileName = os.path.splitext(file)[0]
        fileNames.append(fileName)
  
    print(fileNames)
    print("Images Loaded")
    return images, fileNames

def getEncodes(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    print("Encode Finished")

    return encodeList    

def registerUser(name):
    with open('resources/faces/register.csv','r+') as f:
        dataList = f.readlines()
        nameList = []
        for line in dataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if (name not in nameList):
            now = datetime.now()
            data = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{data}')

def main():

    images, fileNames = readImagesFromDisk();
    encodeImages = getEncodes(images)
    
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()

        if frame is None:
            break

        imgs = cv2.resize(frame,(0,0),None,0.25,0.25)
        imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

        faceCurrentLocation = face_recognition.face_locations(imgs)
        faceCurrentEncode = face_recognition.face_encodings(imgs,faceCurrentLocation)

        for encode, location in zip(faceCurrentEncode, faceCurrentLocation):
            # Verificar si cada una de las imagenes con las que se compara tiene match
            # Array de dimensión el número de images de test que se tiene (4 en este caso)
            matches = face_recognition.compare_faces(encodeImages,encode)
            print(matches)

            # Array de dimensión 4 en este caso indicando la diferencia
            # Número más alto, indica menos parecido. Lo interesante son número bajos
            distance = face_recognition.face_distance(encodeImages,encode)
            print(distance)
            # Se coge la posición de la distancia mínima de las existentes en el array de distancias
            matchIndex = np.argmin(distance)
            y1, x2, y2, x1 = location
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)
            #print(matchIndex)
            if (matches[matchIndex]): # Si el indice del minimo en la distaancia tiene una comparación True
                # Se tiene un match
                name = fileNames[matchIndex].upper()
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                registerUser(name)
                print('He is '+name)



        cv2.imshow("Acceso", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

main()