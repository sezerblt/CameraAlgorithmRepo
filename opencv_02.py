import numpy
import argparse
import imutils
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--camera", required=True,
	help="open camera (Turkce yazmaya gerek yok)")
args = vars(ap.parse_args())

#image = cv2.imread(args["image"])
#cv2.imshow("Image", image[0:500,0:500])
#cv2.waitKey(0)
# convert the image to grayscale
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Gray", gray)
#edged = cv2.Canny(gray, 30, 150)
#cv2.imshow("Edged", edged)
#cv2.waitKey(0)
#"""
dev=args["camera"]
capture = cv2.VideoCapture(int(dev))
while(True):
    ret,frame = capture.read()
    rows,cols,channels =frame.shape
    cv2.line(frame,(10,10),(60,10),(0,100,250),2)
    cv2.line(frame,(10,10),(10,60),(0,100,250),2)
        
    cv2.line(frame,(cols-10,10),(cols-60,10),(0,100,250),2)
    cv2.line(frame,(cols-10,10),(cols-10,60),(0,100,250),2)

    cv2.line(frame,(10,rows-10),(60,rows-10),(0,100,250),2)
    cv2.line(frame,(10,rows-10),(10,rows-60),(0,100,250),2)
        
    cv2.line(frame,(cols-10,rows-10),(cols-10,rows-60),(0,100,250),2)
    cv2.line(frame,(cols-10,rows-10),(cols-60,rows-10),(0,100,250),2)

    cv2.line(frame,(int(cols/2),int(rows/2)-50),(int(cols/2),int(rows/2)+50),(0,0,250),1)
    cv2.line(frame,(int(cols/2)-50,int(rows/2)),(int(cols/2)+50,int(rows/2)),(0,0,250),1)

    cv2.circle(frame,(int(cols/2),int(rows/2)),20,(15,10,250), 2)

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_red=numpy.array([100,40,40])
    upper_red=numpy.array([255,255,150])
        
    mymask = cv2.inRange(hsv,lower_red,upper_red)
    res = cv2.bitwise_and(frame,frame,mask=mymask)
        
    cv2.imshow("Live video",frame)
    cv2.imshow("mask mode",mymask)
    cv2.imshow("res mode",res)
    if cv2.waitKey(1) and 0xFF ==ord('q'):
        break
capture.release()
cv2.destroyAllWindows()

