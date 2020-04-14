import cv2
import numpy

filename = "testimage.jpg"
image = cv2.imread(filename)
#cv2.imshow("Source Image",image)

gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv2.imshow("Gray",gray_image)

gray_image_array = numpy.float32(gray_image)
#(src,blockSize,ksize,k,borderType)
dest_corner =cv2.cornerHarris(gray_image_array,2,3,0.004)
#cv2.imshow("Harris Corner",dest_corner)

#()
mykernel = cv2.getStructuringElement(cv2.MORPH_RECT,ksize=(3,3))
#(src,kernel:Mat,dst,anchor,iterations,borderType,borderValue)
dest_dilate = cv2.dilate(dest_corner,mykernel)
#cv2.imshow("Dilate",dest_dilate)

thresh_val = 0.05*dest_dilate.max()
retv,dest_thresh = cv2.threshold(dest_dilate,thresh_val,255,0)

dest_array=numpy.uint8(dest_thresh)
#(image,labels,stats,centroids,connectivity,ltype)
retv2,labels,stats,centroids = cv2.connectedComponentsWithStats(dest_array)

my_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,100,0.005)
#(image,corners,winsize,zeroZone,criteria)
my_corners = cv2.cornerSubPix(image=gray_image_array,
                                corners=numpy.float32(centroids),
                                winSize=(5,5),
                                zeroZone=(-1,-1),
                                criteria=my_criteria)

result = numpy.hstack((centroids,my_corners))
result2 = numpy.int0(result)
image[ result2[:,1],result2[:,0] ] = [0,0,255]
image[ result2[:,3],result2[:,2] ] = [0,255,0]
cv2.imwrite("test.jpg",image)
cv2.imshow("Result",image)



