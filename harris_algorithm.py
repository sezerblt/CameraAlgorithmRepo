import cv2
import numpy as np

filename="testimage.jpg"
image =cv2.imread(filename)
cv2.imshow("Source Image",image)

gray_image      = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray",gray_image)

gray_input_array= np.float32(gray_image)
corner_harris   = cv2.cornerHarris(src=gray_input_array,blockSize=2,ksize=3,k=0.05)
cv2.imshow("Harris",corner_harris)

myKernel=cv2.getStructuringElement(shape=cv2.MORPH_CROSS,ksize=(3,3))
dilated_dst     = cv2.dilate(src=corner_harris,kernel=myKernel)
cv2.imshow("Dilate",dilated_dst)

image[dilated_dst>0.01*dilated_dst.max()]=[0,0,255]
cv2.imshow("Destination Image",image)
if cv2.waitKey(0) & 0xff==27:
    cv2.destroyAllWindows()
