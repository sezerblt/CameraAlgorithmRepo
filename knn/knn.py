import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

train = np.random.randint(0,100,(30,2)).astype(np.float32)
label = np.random.randint(0,3,(30,1)).astype(np.float32)

knn = cv.ml.KNearest_create()
knn.train(train,cv.ml.ROW_SAMPLE,label)
new_data = np.random.randint(0,100,(10,2)).astype(np.float32)
ret,result,neighbours,dist = knn.findNearest(new_data,5)

red = train[label.ravel()==0]
green = train[label.ravel()==1]
blue =train[label.ravel()==2]

plt.scatter(new_data[:,0],new_data[:,1],(30,),'k','*')
plt.scatter(red[:,0],red[:,1],(40,),'r','o')
plt.scatter(green[:,0],green[:,1],(40,),'g','o')
plt.scatter(blue[:,0],blue[:,1],(40,),'b','o')

print("knn:",new_data)
print("sonuc:  ",result)
print("komsu:  ",neighbours)
print("mesafe: ",dist)
plt.show()
