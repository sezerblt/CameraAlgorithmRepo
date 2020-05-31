import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
X = np.random.randint(25,50,(25,2))
Y = np.random.randint(60,85,(25,2))
Z = np.vstack((X,Y)).astype(np.float32)


# olculeri olustur ve kmeans argumana ele
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv.kmeans(Z,2,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

# ravel metodu ile N.sutunların degerlerini degiskene ata 
A = Z[label.ravel()==0]#0(1.sutun)
B = Z[label.ravel()==1]#1(2.sutun)
#plt ekle
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('yukseklik'),plt.ylabel('genislik')
plt.show()
#-----



