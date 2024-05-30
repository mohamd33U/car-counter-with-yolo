from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("pop.mp4")
mdl = YOLO("yolov8n.pt")
classnames = ['person','bicycle','car','motorcycle','airplane', 'bus','train','truck', 'boat','traffic light',
'fire hydrant','stop sign','parking meter', 'bench','bird','cat','dog','horse', 'sheep', 'cow',
'elephant','bear','zebra','giraffe', 'backpack','umbrella', 'handbag', 'tie','suitcase', 'frisbee'
 ,'skis', 'snowboard', 'sports ball', 'kite','baseball bat','baseball glove', 'skateboard','surfboard',
'tennis racket', 'bottle','wine glass', 'cup', 'fork', 'knife','spoon', 'bowl','banana', 'apple', 'sandwich',
'orange', 'broccoli', 'carrot','hot dog', 'pizza','donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
'dining table', 'toilet','tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
'toaster', 'sink','refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
mask=cv2.imread("mask.jpg")
trkr=Sort(max_age=20,min_hits=3,iou_threshold=0.3)
lmts=[412,487,1172,424]
counts = []

while True:
    _, frame = cap.read()
    imgmask=cv2.bitwise_and(frame,mask)
    #bimg=cv2.imread("ford.jpg",cv2.IMREAD_UNCHANGED)
    img=cvzone.overlayPNG(frame,bimg,(0,0))
    rs = mdl(imgmask, stream=True)
    detcions=np.empty((0,5))
    for r in rs:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls=int(box.cls[0])
            cclass=classnames[cls]
            if cclass=="car"and conf>0.3:
                #cvzone.putTextRect(frame, f'{cclass} {conf}', (max(0, x1), max(35, y1)),scale=0.6,thickness=1,offset=3)
                #cvzone.cornerRect(frame, (x1, y1, w, h), l=9,rt=5)
                carray=np.array([x1,y1,x2,y2,conf])
                detcions=np.vstack((detcions,carray))
    fintks = trkr.update(detcions)
    cv2.line(frame,(lmts[0],lmts[1]),(lmts[2],lmts[3]),(0,55,99),5)
    for fk in fintks:
        x1, y1, x2, y2, id = fk
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(0, 255, 0))
        cvzone.putTextRect(frame, f'{int(id)}',(max(0, x1), max(35, y1)), scale=4, thickness=3, offset=3)
        cx,cy=x1+w//2,y1+h//2
        cv2.circle(frame,(cx,cy),5,(0,255,0),-1)
        if lmts[0]<cx<lmts[2] and lmts[1]-30<cy<lmts[1]+30:
            cv2.line(frame, (lmts[0], lmts[1]), (lmts[2], lmts[3]), (0, 255, ), 5)
            if counts.count(id)==0:
                counts.append(id)

    cvzone.putTextRect(frame, f'count={len(counts)}',(50,50))
    #cvzone.putTextRect(frame, f'count={len(counts)}', (255, 100),cv2.FONT_HERSHEY_PLAIN,5,(12,88,99),8)
    cv2.imshow("ee", frame)
    cv2.waitKey(1)