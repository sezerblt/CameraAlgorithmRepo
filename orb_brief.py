import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

"""
img = cv.imread('c2.jpg',1)

orb = cv.ORB_create()
kp = orb.detect(img,None)
kp, des = orb.compute(img, kp)
img2 = cv.drawKeypoints(img, kp, None, color=(0,0,250), flags=0)
plt.imshow(img2), plt.show()
"""
video =cv.VideoCapture(0)
while True:
    _,frame = video.read()
    #frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    #()
    orb = cv.ORB_create()
    #keyİPoints()
    kp = orb.detect(frame,None)
    #(keypoint,destination)
    kp, des = orb.compute(frame, kp)
    #(image,keypoints,outImage,color,flags)
    frame2 = cv.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)
    
    cv.imshow("result",frame2)
    keypress = cv.waitKey(1) & 0xFF
    if keypress == ord("q"):
        break

video.release()
cv.destroyAllWindow()

