import numpy as np
import cv2 as cv
import argparse
import time

parser = argparse.ArgumentParser(description="Bla bla bla...")
parser.add_argument("--input",type=str,help="Path",default="test.mp4")
parser.add_argument("-algo",type=str,  help="Background-KNN-MOG2",default="MOG2")
args=parser.parse_args()

backSub=None
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubractorKNN()


capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))

if not capture.isOpened:
    print("Erisim Hatali"+args.input)
    print("System Kapatiliyor...")
    time.sleep(1.5)
    
while True:
    grab,Frame = capture.read()
    if Frame is None:
        break

    foreground=backSub.apply(Frame)
    cv.rectangle(Frame,(10,5),(100,50),(200,120,80))
    cv.putText(Frame,str(capture.get(cv.CAP_PROP_POS_FRAMES)),
                         (12,12),
                         cv.FONT_HERSHEY_SIMPLEX,0.5,
                         (150,180,20))
    cv.imshow('Input Data',Frame)
    cv.imshow('ForeGround Mask',foreground)
    key=cv.waitKey(20)
    if key=='q' or key==27:
        break

capture.release()
cv.destroyAllWindows()

    
    
    
