import numpy as np
import cv2 as cv
def nothing(x):
    pass
# Csiyah arayuz
img = np.zeros((300,512,3), np.uint8)
cv.namedWindow('image')
# bar olusturalim
cv.createTrackbar('Red','image',0,255,nothing)
cv.createTrackbar('Green','image',0,255,nothing)
cv.createTrackbar('Blue','image',0,255,nothing)
# switch olusturalim
switch = '0 : OFF \n1 : ON'
cv.createTrackbar(switch, 'image',0,1,nothing)
while(1):
    cv.imshow('image',img)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    # get current positions of four trackbars
    r = cv.getTrackbarPos('R','image')
    g = cv.getTrackbarPos('G','image')
    b = cv.getTrackbarPos('B','image')
    s = cv.getTrackbarPos(switch,'image')
    if s == 0:
        img[:] = 0
    else:
        img[:] = [b,g,r]
cv.destroyAllWindows()
