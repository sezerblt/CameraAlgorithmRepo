import cv2
import numpy as np
import matplotlib.pyplot as plt

image=cv2.imread("test_image.jpg")
image=cv2.resize(image,None,fx=0.5,fy=0.5)

def cannyImage(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(3,3),0)
    canny=cv2.Canny(blur,50,250)
    return canny

def region_interest(image):
    height,width=image.shape[:2]
    """ triangle=np.array([
        [(300,200),(300,height),(750,height)]
        ---------------------------
        triangle=np.array([
        [(371,165),(7,340),(775,340)]
        ]
        ])"""
    triangle=np.array([
        [(int(width/2),int(height/4)),(0,height),(width,height)]
        ])
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,triangle,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,250,0),3)
    return line_image

def make_coordinates(line,line_parameters):
    slope,intercept = line_parameters
    y1=image.shape[0]
    y2=int(y1*(11/20))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])
    
def average_slope_intercept(image,lines):
    print(lines)
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope <0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average=np.average(left_fit,axis=0)
    right_fit_average=np.average(right_fit,axis=0)
    print("left_fit: ",left_fit)
    print("right_fit: ",right_fit)
    print("left_fit_average: ",left_fit_average)
    print("right_fit_average: ",right_fit_average)
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)
    return np.array([left_line,right_line])
    
        #01.09.00

#lane_image=np.copy(image)
#canny_image=cannyImage(lane_image)
#cropped=region_interest(canny_image)

#lines=cv2.HoughLinesP(cropped,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
#average_lines = average_slope_intercept(lane_image,lines)
#line_image =display_lines(lane_image,average_lines)

#combo_image=cv2.addWeighted(lane_image,0.8,line_image,1,1)
 
#cv2.imshow("image",image)
#cv2.imshow("cropped",cropped)
#cv2.imshow("line_image",line_image)
#cv2.imshow("combo_image",combo_image)

capture=cv2.VideoCapture("test_video.mp4")
while(capture.isOpened()):
    ret,frame=capture.read()
    frame=cv2.resize(frame,None,fx=0.5,fy=0.5)
    
    canny_image=cannyImage(frame)
    #cv2.imshow("canny_image",canny_image)
    
    cropped=region_interest(canny_image)
    #cv2.imshow("cropped",cropped)
    
    lines=cv2.HoughLinesP(cropped,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    average_lines = average_slope_intercept(frame,lines)
    line_image =display_lines(frame,average_lines)
    #cv2.imshow("line_image",line_image)

    combo_image=cv2.addWeighted(frame,0.5,line_image,1,1)
    cv2.imshow("combo frame",combo_image)
    k=cv2.waitKey(1)

    if k==27:
        break
    
    

capture.release()
cv2.destroyAllWindows()
#plt.imshow(canny)
#plt.show()

 
