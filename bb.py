import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
img = cv2.imread('pic1.jpeg')
height,width,channel= img.shape
##img[:,:,2]+=20
##mask = np.zeros(img.shape[:2],np.uint8)
##bgdModel = np.zeros((1,65), np.float64)
##fgdModel = np.zeros((1,65), np.float64)
##rect = (0,0,width-1,height-1)
##cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
##mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
##img= img*mask2[:,:,np.newaxis]
w=round(width/3)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1=gray[0:height, 0:w]
img2=gray[0:height, w:2*w]

cv2.imshow("test",img2)
img3=gray[0:height, 2*w:3*w]
cv2.imshow("croped",img1)
dst1=cv2.Canny(img1, 100,100)
#cv2.imshow("win2",dst)
gray = np.float32(dst1)
corners1 = cv2.goodFeaturesToTrack(dst1,200,0.01,10)
corners1 = np.int0(corners1)
dst2=cv2.Canny(img2, 100,100)
#cv2.imshow("win2",dst)
gray2 = np.float32(dst2)
corners2 = cv2.goodFeaturesToTrack(dst2,200,0.01,10)
corners2 = np.int0(corners2)
dst3=cv2.Canny(img3, 100,100)
#cv2.imshow("win2",dst)
gray3 = np.float32(dst3)
corners3 = cv2.goodFeaturesToTrack(dst3,200,0.01,10)
corners3 = np.int0(corners3)
arr=[[]]
arr.remove([])
for corner in corners1:
    x,y = corner.ravel()
    arr.append([x,y])
    cv2.circle(img,(x,y), 3, 255, -1)
pts1 = np.array(arr, np.int32)
pts1 = pts1.reshape((-1,1,2))
cv2.polylines(img,[pts1],True,(255,0,255))
arr=[[]]
arr.remove([])
for corner in corners2:
    x,y = corner.ravel()
    arr.append([w+x,y])
    cv2.circle(img,(w+x,y), 3, 255, -1)
pts2 = np.array(arr, np.int32)
pts2 = pts2.reshape((-1,1,2))
cv2.polylines(img,[pts2],True,(255,255,0))
arr=[[]]
arr.remove([])
for corner in corners3:
    x,y = corner.ravel()
    arr.append([(w*2)+x,y])
    cv2.circle(img,((w*2)+x,y), 3, 255, -1)
pts3 = np.array(arr, np.int32)
pts3 = pts3.reshape((-1,1,2))
cv2.polylines(img,[pts3],True,(0,255,255))
arr=[[]]
arr.remove([])
print(arr)

cv2.imshow('corner',img)
img1 = cv2.imread('pic1.jpeg',0)
img2 = cv2.imread('pic2.jpg',0)

orb=cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

img3= cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], None, flags=2)
small = cv2.resize(img3, (0,0), fx=0.25, fy=0.25) 
cv2.imshow('img3',small)
#plt.show()
