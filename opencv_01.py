import cv2
import imutils
import argparse
import numpy

file="C:\\Users\\rootx\\Desktop\\testler\\car.JPG"
class TestImage():

    def __init__(self,source):
        self.image=cv2.imread(source)
        
        #roi=self.image[10:40,20:50]
        #(B,G,R)=self.image[100,100]
        #print("Blue={}, Green={}, Red={}".format(B,G,R))
        #-----------------------------------------
        
    def getImage(self):
        return self.image

    def details(self):
        (h, w, d) = self.image.shape
        print("heigth(row)={}, widht(col)={}, depth(cha)={}".format(h, w, d))

    def rotationImage(self,image):
        center=(image.shape[0]/2,image.shape[1]/2)
        M=cv2.getRotationMatrix2D(center,-45,1.0)
        rotated=cv2.warpAffine( image,M,(image.shape[0],image.shape[1]) )
        print("rotated image")
        return rotated
    
    def rotationImageWithImutils(self,image):
        rotated =imutils.rotate_bound(image,45)
        print("rotated image with imutils")
        return rotated

    def resizedImage(self,image):
        resized=imutils.resize(image,width=300)#(w,h)
        print("resized image")
        return resized

    def smoothingImage(self):
        smoothing=cv2.GaussianBlur(self.image,(5,5),0)
        print("blurred image")
        return smoothing

    def drawImage(self,mode):
        out=self.image.copy()
        color=(0,50,250)
        pt1=(25,60)
        pt2=(100,150)
        rad=(50,100)
        if mode==0:
            out=cv2.line(out,pt1,pt2,color,2)
        elif mode==1:
            out=cv2.rectangle(out,pt1,pt2,color,2)
        elif mode==2:
            out=cv2.circle(out,rad,30,color,2)
        elif mode==3:
            height=str(self.image.shape[0])
            weight=str(self.image.shape[1])
            text="h:{},w:{}".format(height,weight)
            font=cv2.FONT_HERSHEY_SIMPLEX
            out=cv2.putText(out,text,pt2,font,0.7,color,2)           
        return out


    def cannyEdge(self):
        gray=cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 30, 150)
        return edged

    def thresholdImage(self,ptr):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, ptr, 255, cv2.THRESH_BINARY_INV)[1]
        return thresh

    def detectingContours(self,ptr):
        thresh=self.thresholdImage(ptr)
        cnt=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts=imutils.grab_contours(cnt)
        cnts_img=self.image.copy()
        result=None
        for list_c in cnts:
            cv2.drawContours(cnts_img,[list_c],-1,(20,20,250),1)
        text="Contour number:{}".format(len(cnts))
        pt1=(5,5)
        pt2=(15,15)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cnts_img,text,pt2,font,0.5,(20,200,20),2) 
        result=cnts_img
        return result

    def erosionAndDilationImage(self,ptr,mode):
        mask=self.thresholdImage(ptr)
        if mode=='e' or mode==0:
            new_mask=cv2.erode(mask,None,iterations=5)
        elif mode=='d' or mode==1:
            new_mask=cv2.dilate(mask,None,iterations=5)
        elif mode=='b' or mode==2:
            new_mask=cv2.bitwise_and(self.image,self.image, mask=mask)
        return new_mask

img=TestImage(file)
image=img.getImage()
#draw= img.drawImage(3)
#thresh= img.thresholdImage(127)
#resized = img.resizedImage(self.image)
#rotated = img.rotationImageWithImutils(image)
#blurred = img.smoothingImage()
#dilate=img.erosionAndDilationImage(127,'d')
#erode=img.erosionAndDilationImage(127,'e')
#bitwise=img.erosionAndDilationImage(127,'b')
#cv2.imshow("Dilation", dilate)
#cv2.imshow("Erosion", erode)
#cv2.imshow("Bitwise", bitwise)
cv2.imshow("Image", image)
cv2.waitKey(0)
