import cv2
import numpy as np
cap=cv2.VideoCapture('vid6.mp4')
b=0
x=0
y=0
w=0
h=0
ocenter=[0,0]
cntr=0
while True:
    past=cv2.cvtColor(cap.read()[1],cv2.COLOR_RGB2GRAY)
    present=cv2.cvtColor(cap.read()[1],cv2.COLOR_RGB2GRAY)
    future=cv2.cvtColor(cap.read()[1],cv2.COLOR_RGB2GRAY)
    img1=cv2.absdiff(past,present)
    img2=cv2.absdiff(present,future)
    img3 = cv2.bitwise_and(img1,img2)
    ret,img_fin = cap.read()
    #cv2.imshow('img',img3)
    gray = cv2.cvtColor(img_fin, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    kernel = np.ones((10,10),np.uint8)
    thresh = cv2.dilate(thresh,kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)
    (cnts) = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts[1]:
        if cv2.contourArea(c) < 1000:
            continue
        x, y, w, h = cv2.boundingRect(c)

    #print(str(x))
    x2=x+w
    y2=y+h
    center = [int(round(x2/2)),int(round(y2/2))]
    if not (abs(ocenter[0]-center[0])>20 or abs(ocenter[1]-center[1])>20):
        cv2.line(img_fin,(ocenter[0],ocenter[1]),(center[0],center[1]),(255,0,0),thickness=5)
        cv2.circle(img_fin,(ocenter[0],ocenter[1]),1,(0,255,0))
##        b=5
##    if b==5:
##        break
    cv2.imshow('img',img_fin)
    cv2.imshow('img2',thresh)
    k=cv2.waitKey(30) & 0xff
    cntr+=1
    if k==27:
        break;
    if cntr%3==0:
        ocenter=center
cap.release()   
