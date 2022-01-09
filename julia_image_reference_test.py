#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 20:59:10 2022

@author: softdesert
"""

import cv2
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from pyqtgraph.ptime import time
import pyqtgraph.graphicsItems.ScatterPlotItem
import numpy as np

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.startWindowThread()
video = cv2.VideoCapture(0)
app = pg.mkQApp()
p = pg.PlotWidget()

img = cv2.imread('/volumes/LETS_STORE_SHIT/julia/'+str(int(1*10/640))+'_'+str(int(1*10/480))+'.png',cv2.IMREAD_GRAYSCALE)

while True:    
    check, frame = video.read()
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    #faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
    
    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
        print(int(xA*100/640),int(yA*100/480))
        img = cv2.imread('/volumes/LETS_STORE_SHIT/julia/'+str(int(xA*100/640))+'_'+str(int(yA*10/480))+'.png',cv2.IMREAD_GRAYSCALE)
    # for x,y,w,h in faces:
    #     frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)
    #     print(int(x*100/640),int(y*100/480))
    #     img = cv2.imread('/volumes/LETS_STORE_SHIT/julia/'+str(int(x*100/640))+'_'+str(int(y*10/480))+'.png',cv2.IMREAD_GRAYSCALE)
    #     ret,frame = cv2.threshold(frame,80,255,cv2.THRESH_BINARY)
        
    cv2.imshow("fractal",img)
    cv2.imshow('human detector', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break;
video.release()
cv2.destroyAllWindows()