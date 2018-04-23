import  cv2
import numpy as np
cap = cv2.VideoCapture('vid3.mp4')
while True:
    f = open("a1.txt","r")
    
    run_var = f.readline()
    f.close()
    ret,frame = cap.read()
    cv2.imshow('frame',frame)
    if run_var == "1":
     
       cv2.imwrite('test_images/image1.jpg',frame)
     
