import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img='C:\\Users\\rootx\\Downloads\\logos\\classic.JPG'
image = cv.imread(img)
def getBGR(image):
    blue,red,green=cv.split(image)
    new_image=cv.merge((blue,green,red))
    return blue,red,green

def getSingleColr(image,x,ych):
    channel = bgr=image[x,y,val]
    return channel

def getAllSingleColr(image,ch):
    channel = image[:,:,val]
    return channel
def getColorList(image,x,y):
    px_color=image[120,150]
    return px_color

def borderIm(img1):
    BLUE=[0,250,0]
    replicate = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REPLICATE)
    reflect = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT)
    reflect101 = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT_101)
    wrap = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_WRAP)
    constant= cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_CONSTANT,value=BLUE)
    plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
    plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
    plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
    plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
    plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
    plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
    plt.show()
    
borderIm(image)
