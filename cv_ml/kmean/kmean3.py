import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img=cv.imread('rgb.PNG')
I=img.reshape((-1,3))
I=np.float32(I)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret2,label2,center2=cv.kmeans(I,8,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

center2=np.uint8(center2)
result=center2[label2.flatten()]
result2=result.reshape((img.shape))

cv.imshow("",result2)
cv.waitKey(0)
cv.destroyAllWindows()
