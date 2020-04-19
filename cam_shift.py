import numpy as np
import cv2 as cv
import argparse

"""
parser = argparse.ArgumentParser()
parser.add_argument('')
args = parser.parse_args()
"""
cap = cv.VideoCapture(0)

ret,frame = cap.read()

x, y, w, h = 300, 200, 100, 50 #
track_window = (x, y, w, h)
track_window2 = (x, y, w, h)
roi = frame[y:y+h, x:x+w]
#ndarray <-(sorce,color space code,destintion channel image)
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
#ndarray <- (src.input array,lower boundary,upper boundary,destination)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
#ndarray <-(images[],channes[i],mask_ndarray,histsize[],range[],accumulate_bool)
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
#
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
#
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
while(1):
    ret, frame = cap.read()
    frame =cv.flip(frame,1)
    frame2=frame.copy()
    ret2=False
    if ret == True:
        ret2==ret
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # 
        ret, track_window = cv.CamShift(dst, track_window, term_crit)
        ret2, track_window2 = cv.meanShift(dst,track_window2,term_crit)
        
        image2 = cv.rectangle(frame,(x,y),(x+w,y+h),(250,0,0),2)
        cv.rectangle(frame,(x+10,y+10),(x+w+15,y+h+15),(0,250,0),2)
        cv.image = cv.putText(frame, 'test', (x-10,y-10),
                               cv.FONT_HERSHEY_SIMPLEX,1,  
                   (0,0,250),1)
        cv.imshow("image",image2)
        # 
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv.polylines(frame,[pts],True, 255,2)
        cv.imshow('img2',img2)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break


cap.release()
cv2.destroyallWindows()
