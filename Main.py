import os
import numpy as np
import tensorflow_hub as hub
import time
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from Resource.FunctionUtils import Standard_colors, WriteResult
from DataMemory import TrackingMemory

#GPU settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Open Video_input from Resource
source_video='Resource/VideoInput/input3.mp4'
cap = cv2.VideoCapture(source_video)

#Take The label_path from Resource
PATH_TO_LABELS = 'Resource/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

#Open and Load Models of neural network that we use
urlDetector="Models/SSD_MODELS"
#urlDetector="Models/CENTERNET_MODELS"
detector = hub.load(urlDetector)
print("Model Load")

#VIDEOWRITER
#Open and settings videowriter
width=640
height=480
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#take name of models
nameDetector=urlDetector.split("/")[-1]+'.avi'
#name of outputVideo
nameOutputVideo=source_video.split("/")[-1].split(".")[0]+"_"+nameDetector+".mp4"
#open writer
output_movie = cv2.VideoWriter("output/"+nameOutputVideo, fourcc, fps, (width,height))
print("name detector ",nameDetector)

#initialize memory struct for data
counter=0
DataMemory=[]
#import colors from function
colors=Standard_colors()

while cap.isOpened():
    
    #read frame from video
    ret, frame = cap.read()
    if not ret:
        print ('end of the video file...')
        break

    #increment counter of frame
    counter=counter+1   

    #resize frame for work on it
    frame=cv2.resize(frame, (width,height))
    frame_exp = np.expand_dims(frame, axis=0)

    #apply model on image, we can count how long does it take to analize image
    #start_time = time.monotonic()
    results =detector(frame_exp)
    #print('seconds: ', time.monotonic() - start_time)
    result = {key:value.numpy() for key,value in results.items()}
    
    #we study only vehicles, i take from label_path the id for car, truck, motorbike, bus
    UsefulClass=[3,4,6,8]
    Boxes=[]
    Classes=[]
    Center=[]
    #select from the model's result only the prediction with superior accuracy of 50% and right id 
    for i in range(len(result['detection_boxes'][0])):
      if result['detection_scores'][0][i] > .5 and result['detection_classes'][0][i] in UsefulClass :
        #save for all of these: Boxe contains prediction, Id class, Center
        Boxes.append(result['detection_boxes'][0][i])
        Classes.append(result['detection_classes'][0][i])
        Center.append((((Boxes[-1][1]+Boxes[-1][3])/2.0)*width,((Boxes[-1][0]+Boxes[-1][2])/2.0)*height))
    
    exPrevision=[]
    #if i find some of useful i proceede
    if(bool(Center)):
      #map the center how integer and not float
      Center=tuple(tuple(map(int, tup)) for tup in Center)

      for i in range(0,len(Center)):
       
        #SET CONSTANT
        DistMin=80 
        #call function of tracking
        DataMemory,exPrevision=TrackingMemory(DataMemory,Classes[i],Center[i],Boxes[i],counter,DistMin)

      #if i have expection from tracking, drawing the prediction of that
      for i in range(0,len(exPrevision)):
        cv2.circle(frame,exPrevision[i][0], radius=3, color=(255, 255, 255), thickness=6)  #Center
        cv2.circle(frame,exPrevision[i][1], radius=5, color=(0, 0, 0), thickness=6)  #Pline
        cv2.line(frame, exPrevision[i][0],exPrevision[i][1], (0, 0xFF, 0), 5)  #Center-Pline
        cv2.line(frame, exPrevision[i][1],exPrevision[i][2], (0, 0xFF, 0xFF), 5)  #Pline-DataCente
        #input("stop")
        


    #draw result of tracking
    for l in range(0,len(DataMemory)):
      #divide the boxe
      ymin, xmin, ymax, xmax = DataMemory[l][5][0]
      #create ticket over the image
      TickName= [str(category_index[DataMemory[l][1][0]]['name'])+" "+str(DataMemory[l][0]), str("velocity "+ str(DataMemory[l][4][2]))]

      #want to draw only the prediction's image of that frame
      if DataMemory[l][2][0]==counter:
        viz_utils.draw_bounding_box_on_image_array(frame_exp[0], ymin, xmin, ymax , xmax, color=colors[l], thickness=2, display_str_list=TickName, use_normalized_coordinates=True) #boxe
        cv2.circle(frame,DataMemory[l][3][0], radius=3, color=(0xFF, 0, 0), thickness=3)   #center

    #write video
    frame=cv2.resize(frame, (width,height))
    output_movie.write(frame)
    
    #show frame
    cv2.imshow('object detection',frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break

#print the result of trakink
for i in range(0,len(DataMemory)):
  print("data ",DataMemory[i])

#create txt file with result
nameTxt = "output/"+source_video.split("/")[-1]+"_"+urlDetector.split("/")[-1]+".txt"
WriteResult(nameTxt, DataMemory, category_index)

cap.release()
cv2.destroyAllWindows()
