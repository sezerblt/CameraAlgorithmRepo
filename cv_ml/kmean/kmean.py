import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
x = np.random.randint(25,100,20)
y = np.random.randint(175,200,20)
z = np.hstack((x,y))
z = z.reshape((len(x)+len(y),1))
z = np.float32(z)


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

flags = cv.KMEANS_RANDOM_CENTERS
#
compactness,labels,centers = cv.kmeans(z,2,None,criteria,10,flags)
# ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# flags=2
flags = cv.KMEANS_RANDOM_CENTERS
# Apply KMeans
compactness,labels,centers = cv.kmeans(z,2,None,criteria,10,flags)
A = z[labels==0]
B = z[labels==1]
plt.hist(A,256,[0,256],color = 'r')
plt.hist(B,256,[0,256],color = 'b')
plt.hist(centers,32,[0,256],color = 'y')
plt.show()
