import cv2 as cv
import numpy as np


cap = cv.VideoCapture(0)
ret,frame = cap.read()
#frame=cv.flip(frame,1)
x, y, w, h = 300, 200, 100, 50 
track_window = (x, y, w, h)
roi = frame[y:y+h, x:x+w]

hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

term_crit = ( cv.TERM_CRITERIA_EPS, 10, 1 )

while(1):
    ret, frame = cap.read()
    frame=cv.flip(frame,1)
    if ret == True:
        red=(0,0,255)
        green=(0,255,0)
        blue=(255,0,0)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # meanshift
        ret, track_window = cv.meanShift(dst, track_window, term_crit)
        x,y,w,h = track_window
        #cv.line(frame,(x+w//2,y+h//2),(x+w//2,y+h),(0,0,255),2)
        #cv.line(frame,(x,y+h//2),(x+w//2,y+h//2),(0,0,255),2)
        """
        img2=cv.rectangle(frame, (x+30,y-30), (x+w+30,y+h-30),red ,2)
        cv.line(frame,(x+30,y-30),(x,y),green,2)
        cv.line(frame,(x+w+30,y-30),(x+w,y),green,2)
        
        img2 = cv.rectangle(frame, (x,y), (x+w,y+h), blue,2)
        cv.line(frame,(x,y+h),(x+30,y+h-30),green,2)
        cv.line(frame,(x+w,y+h),(x+w+30,y+h-30),green,2)
        """
        cv.line(frame,(x,y),(x,y+15),green,2)
        cv.line(frame,(x,y),(x+15,y),green,2)
        
        cv.line(frame,(x+w,y),(x+w-15,y),red,2)
        cv.line(frame,(x+w,y),(x+w,y+15),red,2)
        
        cv.line(frame,(x,y+h),(x,y+h-15),blue,2)
        cv.line(frame,(x,y+h),(x+15,y+h),blue,2)
        
        cv.line(frame,(x+w,y+h),(x+w-15,y+h),green,2)
        cv.line(frame,(x+w,y+h),(x+w,y+h-15),green,2)
        print(frame)
        
        cv.imshow('img2',frame)
        k = cv.waitKey(30) & 0xFF
        if k == 27:
            break
    else:
        break
