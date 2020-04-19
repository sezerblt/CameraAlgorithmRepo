#R=L1*L2-k*(L1+L2)^2
#R=min(L1,L2)
import numpy
import cv2
from matplotlib import pyplot

video =cv2.VideoCapture(0)
count=0
res=False
while True:
    grab,frame=video.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #(image,maxCorners,qualityLevel,minDistance,mask,blockSize,gradientSize,corners,useHarrisDetector)
    corners = cv2.goodFeaturesToTrack(gray,25,0.05,10)
    corners_array=numpy.int0(corners)
    
    for i in corners_array:
        # [[],[],[],...,[]]
        # matrisi  diziye cözümle (x,y)
        x_point,y_point = i.ravel()
        cv2.circle(frame,(x_point,y_point),8,(200,100,0),2)
    
    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord("q"):
        break

    cv2.imshow('Result Point',frame)
    #pyplot.imshow(frame)
    #pyplot.show()
video.release()
cv2.destroyAllWindows()
