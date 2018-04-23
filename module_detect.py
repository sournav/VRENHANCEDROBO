


# # Speed Limit Detection Demo

f = open("a1.txt","w")

f.write("0"+"\n")
f.close()
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image,ImageEnhance,ImageFilter
import pytesseract
import dmodule
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
if tf.__version__ != '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')


sys.path.append("..")


from utils import label_map_util

from utils import visualization_utils as vis_util





# What model to use.
MODEL_NAME = 'module_graph'


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object_detection.pbtxt')

NUM_CLASSES = 1





detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')





label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)




def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)




# Add path to images directory below
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


corners2x=[]
corners2y=[]
it=0
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    i=1
    target=[]
    target2=[]
    for image_path in TEST_IMAGE_PATHS:
      
      image = Image.open(image_path)
      img = cv2.imread(image_path)
      height, width, channels = img.shape
    #cap = cv2.VideoCapture('vid6.mp4')
    #fgbg = cv2.createBackgroundSubtractorMOG2()
    #import motion_detect_module 
    
      image_np = load_image_into_numpy_array(image)
      #height,width,channel = image_np.shape
      
      #fgmask = fgbg.apply(image_np)
      #cv2.imshow('fg',fgmask)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      #finding which box is the bounding box
      y1=0
      y2=0
      x1=0
      x2=0
      boxy=np.squeeze(boxes)
      score=np.squeeze(scores)
    ##      classif = np.squeeze(classes)
      
      detections = []
      for i in range(boxy.shape[0]):
          if score[i]>=0.3:
            y1,x1,y2,x2=tuple(boxy[i].tolist())
            y1=int(round(y1*height))
            x1=int(round(x1*width))
            y2=int(round(y2*height))
            x2=int(round(x2*width))
            detections.append([y1,x1,y2,x2])
            
  ##          #print(str(x1)+","+str(x2)+","+str(width))
  ##          #print(str(x1)+","+str(y1)+","+str(x3)+","+str(x4))
  ##  ##            classx = str(classif[i])
  ##  ##            f = open("a1.txt","a")
  ##  ##            f.write(str(x1)+","+str(y1)+","+str(x2)+","+str(y2)+","+str(score[i])+","+classx+"\n")
  ##  ##            f.close()
  ##    #print("("+str(x1)+","+str(x2)+","+str(width)+")")
      #x=len(detections)
      y=0
      
      for i in detections:
        
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        y1=i[0]
        x1=i[1]
        y2=i[2]
        x2=i[3]
        ##gray = cv2.cvtColor(img_fin, cv2.COLOR_BGR2GRAY)
        ret, thresh2 = cv2.threshold(gray, 127, 255, 0)
        kernel = np.ones((10,10),np.uint8)
        thresh = cv2.dilate(thresh2,kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh2,cv2.MORPH_CLOSE,kernel)
        gray = np.float32(gray)
        ret, thresh = cv2.threshold(gray[y1:y2,x1:x2], 127, 255, 0)
        kernel = np.ones((10,10),np.uint8)
        thresh = cv2.dilate(thresh,kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)
        gray = np.float32(thresh)
        new_image = image_np[y1:y2,x1:x2]*thresh[:,:,None].astype(image_np.dtype)
        new_image = cv2.cvtColor(new_image,cv2.COLOR_BGR2RGB)
        cv2.imshow('bob'+str(y),new_image)
        #print(gray.shape)
        corners1 = cv2.goodFeaturesToTrack(gray,200,0.01,10)
        corners1 = np.int0(corners1)
        _,cnts,_ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        thresh = cv2.convertScaleAbs(thresh)
        cv2.imshow('thresh'+str(y),thresh)
        y+=1
        
        
	# loop over the contours
        for c in cnts:
          if cv2.contourArea(c)<50:
            continue
          
	  # if the contour is too small, ignore it
##	  if cv2.contourArea(c) < 50:
##	      continue
 
          # compute the bounding box for the contour, draw it on the frame,
          # and update the text
          (xb1, yb1, wb1, hb1) = cv2.boundingRect(c)
          #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
          target2.append(hb1)
        print(str(x1),str(y1),str(x2),str(y2))
        if x1>160 and y1>200 and x2<350 and y2<650:
          target.append([x1,y1,x2,y2])
  
      #im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
      #cnt = contours[4]
      #cv2.drawContours(image_np, [cnt], 0, (0,255,0), 3)
      #i=0
      #for corner in corners1:
      #  x,y = corner.ravel()
        
    ##        if it==1:
    ##           tempx=x-corners2x[i-1]
  ##  ##           tempy=y-corners2y[i-1]
  ##  ##           cv2.line(image_np,(x1+x,y1+y),(x1+corners2x[i-1],y1+corners2y[i-1]),(255,0,0),5)
  ##      cv2.circle(image_np,(x1+x,y1+y), 3, 255, -1)
  ##    #corners2=corners1
  ##  ##      new_image = cv2.bitwise_and(gray[y1:y2,x1:x2],thresh)
  ##  ##      cv2.imshow(new_image)
  ##  ##      #cropping the image to the box
  ##  ##      im=Image.open(image_path)
  ##  ##      im.save("pic2.jpeg","jpeg")
  ##  ##      im=im.crop((x1,y1,x2,y2))
  ##  ##      #plt.figure(figsize=IMAGE_SIZE)
  ##  ##      #plt.imshow(im)
  ##  ####      enhancer = ImageEnhance.Brightness(im)
  ##  ####      factor=3.0
  ##  ##      #im=enhancer.enhance(factor)
  ##  ##      
  ##  ##      #im=im.filter(ImageFilter.SMOOTH)
  ##  ##      
  ##  ##      
  ##  ##      im.save("pic1.jpeg","jpeg")
  ##  ####      im2=cv2.imread("pic1.jpeg")
  ##  ####      im1 = cv2.imread("pic1.jpeg")
  ##  ####      th, dst=cv2.threshold(im2, 100,255, cv2.THRESH_BINARY)
  ##  ####      dst=cv2.Canny(im2, 100,100)
  ##  ####      kernel2 = np.ones((25,25),np.uint8)
  ##  ####      kernel1 = np.ones((1,1),np.uint8)
  ##  ####      erosion = cv2.erode(dst,kernel1,iterations =1)
  ##  ####      #dilate = cv2.dilate(erosion,kernel2,iterations=1)
  ##  ####      
  ##  ####      opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel1)
  ##  ####      closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
  ##  ####      height,width,_ = im1.shape
  ##  ####     
  ##  ####      mask = np.zeros(im1.shape[:2],np.uint8)
  ##  ####      bgdModel = np.zeros((1,65), np.float64)
  ##  ####      fgdModel = np.zeros((1,65), np.float64)
  ##  ####      rect = (0,0,width-1,height-1)
  ##  ####      cv2.grabCut(im1,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
  ##  ####      mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
  ##  ####      im1= im1*mask2[:,:,np.newaxis]
  ##      
  ##      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
      cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
  out=dmodule.work(target[0],target[1],target2[0],target2[1])
  print(str(out))
        #if cv2.waitKey(25) & 0xFF == ord('q'):
          #cv2.destroyAllWindows()
          #break
  ##      f = open("a1.txt")
  ##      lines = f.read().splitlines()
  ##      f.close()
  ##      lines[0] = "1"
  ##      f = open("a1.txt",'w')
  ##      f.write('\n'.join(lines))
  ##      f.close()
        #plt.figure(figsize=IMAGE_SIZE)
        #plt.imshow(image_np)
        #cv2.imshow("detected",image_np)
        #cv2.imshow("assigned",im1)
        #cv2.imshow("cropped",im2)
        #plt.figure(figsize=IMAGE_SIZE)
        #plt.imshow(im)
        
        


