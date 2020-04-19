import numpy
import cv2
import argparse
import time
#komut icin gerekli parse degerleri olustur
"""
parser=argparse.AugmentParser()
parser.add_argument()
args=parser.parse_args()
"""
#görüntünün degerini tututan isaretcimiz
capture = cv2.VideoCapture(0)

#ilk test görüntümüzü okuyalım
ret,frame = capture.read()
x_pos,y_pos,widht,height = 320,240,100,60
track_window = (x_pos,y_pos,widht,height)

#parcali bolumu ekleyelim
roi = frame[y_pos:y_pos+height, x_pos:x_pos+widht]
hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
#mask <- (source,lower_boundary,upper_boundary,dest)
mask = cv2.inRange(hsv_roi,numpy.array((0,64,32)),numpy.array((60,255,185)))
#hist <- (images,channels,mask,histSize,ranges,hist,accumulate)
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# dest <- (src,dest,alpha,beta,norm_Type,dtype,mask)
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

mode=False
#
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)

while True:
    ret,frame = capture.read()
    frame =cv2.flip(frame,1)
    if ret==True:
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        dst =cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        ret,track_window = cv2.meanShift(dst,track_window,term_criteria)
        x_pos,y_pos,widht,height = track_window

        cv2.rectangle(frame,(x_pos+10,y_pos+10),(x_pos+widht+10,y_pos+height+10),(20,200,20),2)
        cv2.putText(frame, 'Test ediliyor...', (x_pos+5,y_pos-10),
                    cv2.FONT_HERSHEY_SIMPLEX ,  
                   1,(20,20,220), 1) 
        image2 = cv2.rectangle(frame,(x_pos,y_pos),(x_pos+widht,y_pos+height),250,2)
        cv2.imshow("image",image2)

        key= cv2.waitKey(30) & 0xff
        if key == 27:
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()
