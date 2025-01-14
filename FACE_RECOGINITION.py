import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = "your file path"
images = []
classNames = []
myList = os.listdir(path)
#print(myList)

for cls in myList:
    currentImg = cv2.imread(f'{PATH}/{cls}') #Replace PATH with location of IMG folder
    images.append(currentImg)
    classNames.append(os.path.splitext(cls)[0])
#print(classNames)

def FindEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def DetectedList(name):
    with open("PATH\Detected.csv",'r+') as f: #ADD location for csv file to be saved
        DataList = f.readline()
        nameList = []
        for line in DataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



encodeListKnown = FindEncodings(images)
print("Encoding completed")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    CurrentFace = face_recognition.face_locations(imgS)
    CurrentFaceEncode = face_recognition.face_encodings(imgS,CurrentFace)

    for encodeFace,faceloc in zip(CurrentFaceEncode,CurrentFace):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        FaceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(FaceDis)
        matchInx = np.argmin(FaceDis)

        if matches[matchInx]:
            name = classNames[matchInx].upper()
            #print(name)

            y1,x2,y2,x1 = faceloc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(102,255,178),2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            DetectedList(name)

    cv2.imshow('Webcam',img)
    key = cv2.waitKey(1)
    if key == 1:
        break

