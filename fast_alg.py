import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


count=0
video =cv.VideoCapture(0)
while True:
    count +=1
    grab,frame = video.read()
    
    # 
    fast = cv.FastFeatureDetector_create()
    # 
    kp = fast.detect(frame,None)
    frame2 = cv.drawKeypoints(frame, kp, None, color=(200,150,90))
    # 
    print( "Threshold deger: {}".format(fast.getThreshold()) )
    print( "nonmaxSuppression fast :{}".format(fast.getNonmaxSuppression()) )
    print( "neighborhood deger: {}".format(fast.getType()) )
    print( "nonmaxSuppression key: {}".format(len(kp)) )

    if count== 120:
        cv.imwrite('default.png',frame2)
        print("{}.sn kayit alindi".format(count))
    cv.imshow('Fast_TRUE',frame2)
    
    fast.setNonmaxSuppression(0)
    kp = fast.detect(frame,None)
    print( "onmaxSuppression key: {}".format(len(kp)) )

    frame3 = cv.drawKeypoints(frame, kp, None, color=(25,180,120))
    if count== 240:
        cv.imwrite('FALSE.png',frame3)
        print("{}.sn kayit alindi".format(count))
    

    keypress = cv.waitKey(1) & 0xFF
    if keypress == ord("q"):
        break

    cv.imshow('Result Output',frame)

video.release()
cv.destroyAllWindows()
