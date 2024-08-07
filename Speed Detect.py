import streamlit as st
import cv2
from tracker2 import *
import numpy as np
end = 0


st.title("Detection of Traffic Rules Violation")
# uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
# for uploaded_file in uploaded_files:
#     bytes_data = uploaded_file.read()
#     st.write("filename:", uploaded_file.name)
#     st.write(bytes_data)

    
#Trackes Object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("Dataset/traffic4.mp4")
f = 25
w = int(1000/(f-1))


#Object Detection
object_detector = cv2.createBackgroundSubtractorMOG2(history=None,varThreshold=None)
#100,5

#KERNALS
#kernalOp = np.ones((1,1),np.uint8)
kernalOp = np.ones((3,3),np.uint8) #for traffic4
#kernalOp2 = np.ones((3,3),np.uint8)
kernalOp2 = np.ones((5,5),np.uint8) #for traffic4
#kernalCl = np.ones((7,7),np.uint8)
kernalCl = np.ones((11,11),np.uint8) #for traffic4
fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=True)
#kernal_e = np.ones((17,17),np.uint8)
kernal_e = np.ones((5,5),np.uint8) #for traffic4

while True:
    ret,frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    height,width,_ = frame.shape
    #print(height,width)
    #540,960


    #Extract ROI
    #roi = frame[10:540,200:960] 
    roi = frame[50:540,200:960] #traffic4
 

    #MASKING METHOD   
    fgmask = fgbg.apply(roi)
    #ret, imBin = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY) #for source
    ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY) #for traffic4
    mask1 = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp)
    mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernalCl)
    e_img = cv2.erode(mask2, kernal_e)


    contours,_ = cv2.findContours(e_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        #THRESHOLD
        #if area > 5000: #for source
        if area > 1000: #for traffic4
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)
            detections.append([x,y,w,h])

    #Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x,y,w,h,id = box_id


        if(tracker.getsp(id)<tracker.limit()):
            cv2.putText(roi,str(id)+" "+str(tracker.getsp(id)),(x,y-15), cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
        else:
            cv2.putText(roi,str(id)+ " "+str(tracker.getsp(id)),(x, y-15),cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 255),2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 165, 255), 3)

        s = tracker.getsp(id)
        if (tracker.f[id] == 1 and s != 0):
            tracker.capture(roi, x, y, h, w, s, id)

    # DRAW LINES

    cv2.line(roi, (0, 410), (960, 410), (0, 0, 255), 2)
    cv2.line(roi, (0, 430), (960, 430), (0, 0, 255), 2)

    #cv2.line(roi, (0, 180), (960, 180), (0, 0, 255), 2)  #for source
    cv2.line(roi, (0, 235), (960, 235), (0, 0, 255), 2) #for traffic4
    #cv2.line(roi, (0, 200), (960, 200), (0, 0, 255), 2)  #for source
    cv2.line(roi, (0, 255), (960, 255), (0, 0, 255), 2)  #for traffic4


    #DISPLAY
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(w-10)
    if key==27:
        tracker.end()
        end=1
        break

if(end!=1):
    tracker.end() #closes implementation

cap.release()
cv2.destroyAllWindows()
