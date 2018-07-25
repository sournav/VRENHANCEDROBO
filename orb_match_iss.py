"""
MADE BY SOURNAV SEKHAR BHATTACHARYA
VR ENHANCED ROBOTICS TEAM 2018
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import threading
# Initiate SIFT detector
def work(cam_file,module):
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
    image_manifest = open(cam_file+'/manifest.txt','r')
    lines = image_manifest.read().splitlines()
    size = len(lines)
    third=round(size/3)
    file = ""
    name = ""
    part = ()
    img2 = cv2.imread(cam_file+'/'+module,0) # trainImage
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    good = [[]]
    leng =[]
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches=[]
    
    def file_thread( start, end ):
        i=start
        while i<end:
            part=lines[i].partition(" ")
            file =part[0]
            name =part[2]
            names1 = name.partition(" ")
            name = names1[0]
            pointxy = names1[2].partition(",")
            pointx.append(pointxy[0])
            pointy.append(pointxy[2])
            img=cv2.imread(cam_file+'/'+file,0)          # queryImage
            names.append(name)
            kp1, des1 = sift.detectAndCompute(img,None)
            matches.append(flann.knnMatch(des1,des2,k=2))
            x=[0]*size
            for m,n in matches[i]:
                if m.distance < 0.7*n.distance:
                    #good[i].append(m)
                    x[i]+=1
                #good.append([])
            leng.append(x[i])
            
            #kp.append(kp1)
            #des.append(des1)
            i+=1
    def third_one():
        file_thread(0,third)
    def third_two():
        file_thread(third,2*third)
    def third_three():
        file_thread(2*third,size)
    
    thread1= threading.Thread(target=third_one())
    thread2= threading.Thread(target=third_two())
    thread3= threading.Thread(target=third_three())
    thread1.start()
    thread2.start()
    thread3.start()
    



    # find the keypoints and descriptors with SIFT


        #for i in range(size):
        

    # store all the good matches as per Lowe's ratio test.
    
    #for i in range(size):
    
    maxval=leng[0]
    maxind=0
    it=1
    while it<size:
        if leng[it]>maxval:
            maxval=leng[it]
            maxind=it
        it+=1
    ret_name=names[maxind]
    ret_x=pointx[maxind]
    ret_y=pointy[maxind]
    return int(ret_x),int(ret_y),ret_name

#x,y,name=work()
#print(name)
