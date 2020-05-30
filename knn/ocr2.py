import cv2 as cv
import numpy as np

mydata = np.loadtxt("letter-recognition.data",
                    dtype="float32",delimiter=',',
                   converters={
                       0:lambda ch:ord(ch)-ord('A')
                       }
                    )
train,test = np.vsplit(mydata,2)
responses,train_data = np.hsplit(train,[1])
labels,test_data = np.hsplit(test,[1])

knn=cv.ml.KNearest_create()
knn.train(train_data,cv.ml.ROW_SAMPLE,responses)
ret,result,neighbour,distance = knn.findNearest(test_data,5)

correct = np.count_nonzero(result==labels)
accuracy = correct*100.0/10000
print(accuracy)

