import cv2
import numpy as np
f = open("a1.txt","r")

run_var = f.readline()

while True:
    f = open("a1.txt","r")

    run_var = f.readline()
    f.close()
    if run_var == "1":
        break
    
im2=cv2.imread("pic1.jpeg")
im1 = cv2.imread("pic1.jpeg")
th, dst=cv2.threshold(im2, 100,255, cv2.THRESH_BINARY)
dst=cv2.Canny(im2, 100,100)
kernel2 = np.ones((25,25),np.uint8)
kernel1 = np.ones((1,1),np.uint8)
erosion = cv2.erode(dst,kernel1,iterations =1)
#dilate = cv2.dilate(erosion,kernel2,iterations=1)

opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel1)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
height,width,_ = im1.shape

mask = np.zeros(im1.shape[:2],np.uint8)
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)
rect = (0,0,width-1,height-1)
cv2.grabCut(im1,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
im1= im1*mask2[:,:,np.newaxis]
cv2.imshow("assigned",im1)
cv2.imshow("cropped",im2)
    
