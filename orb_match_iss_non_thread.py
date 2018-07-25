"""
MADE BY SOURNAV SEKHAR BHATTACHARYA
VR ENHANCED ROBOTICS TEAM 2018
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
# Initiate SIFT detector
def work(cam_folder,test):
    ret_name=''
    ret_x=0
    ret_y=0
    sift = cv2.xfeatures2d.SIFT_create()
    #screen=cv2.imread('screen.jpg',0)
    MIN_MATCH_COUNT = 10
    img =[]
    names =[]
    kp=[]
    des=[]
    pointx=[]
    pointy=[]
    image_manifest = open(cam_folder+'/manifest.txt','r')
    lines = image_manifest.read().splitlines()
    size = len(lines)
    file = ""
    name = ""
    part = ()
    for i in range(size):
        part=lines[i].partition(" ")
        file =part[0]
        name =part[2]
        names1 = name.partition(" ")
        name = names1[0]
    ##    point = part[3]
        pointxy = names1[2].partition(",")
        pointx.append(pointxy[0])
        pointy.append(pointxy[2])
        img.append(cv2.imread(cam_folder+'/'+file,0) )         # queryImage
        names.append(name)
        kp1, des1 = sift.detectAndCompute(img[i],None)
        kp.append(kp1)
        des.append(des1)
    img2 = cv2.imread(cam_folder+'/'+test,0) # trainImage



    # find the keypoints and descriptors with SIFT


    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches=[]
    for i in range(size):
        matches.append(flann.knnMatch(des[i],des2,k=2))

    # store all the good matches as per Lowe's ratio test.
    
    leng =[]
    x=[0]*size
    for i in range(size):
        for m,n in matches[i]:
            if m.distance < 0.7*n.distance:
                #good[i].append(m)
                x[i]+=1
            
        #print(len(good))
        leng.append(x[i])
    result = max(leng)
    #print(len(pointx))
    for i in range(size):
        if result == leng[i]:
            ret_name=names[i]
            ret_x=pointx[i]
            ret_y=pointy[i]
    return ret_x,ret_y,ret_name
        
#x,y,name=work()
#print(name)
##
