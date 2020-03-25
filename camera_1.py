import cv2
import numpy as np
from matplotlib import pyplot as plt
#-------------------------------------------
src="C:\\Users\\rootx\\Desktop\\testler\\car.JPG"
src2="C:\\Users\\rootx\\Desktop\\face1.JPG"
#--------------------------------------------

def quitFrame():
    cv2.destroyAllWindows()
    
def readImage(path,flag):
    img = cv2.imread(path,flag)
    cv2.imshow("outputImage",img)
    cv2.waitKey(0)
    quitFrame()
    return img

readImage(src,0)

def writeImage(src_path,dest_path,flag):
    img = cv2.imread(src_path,flag);
    cv2.imshow("save(s) press",img)
    key = cv2.waitKey(0)
    if(key==27):#press ESC(27)
        quitFrame()
    elif(key==ord('s')):#save
        cv2.imwrite(dest_path,img)
        quitFrame()


def showPlot(src):
    img=cv2.imread(src,cv2.IMREAD_GRAYSCALE)
    plt.imshow(img,cmap='gray',interpolation='bicubic')
    plt.xticks([]),plt.yticks([])
    plt.show()
    
def openCamera(dev):
    capture = cv2.VideoCapture(dev)
    while(True):
        ret,frame = capture.read()
        rows,cols,channels =frame.shape
        cv2.line(frame,(10,10),(60,10),(0,100,250),2)
        cv2.line(frame,(10,10),(10,60),(0,100,250),2)
        
        cv2.line(frame,(cols-10,10),(cols-60,10),(0,100,250),2)
        cv2.line(frame,(cols-10,10),(cols-10,60),(0,100,250),2)

        cv2.line(frame,(10,rows-10),(60,rows-10),(0,100,250),2)
        cv2.line(frame,(10,rows-10),(10,rows-60),(0,100,250),2)
        
        cv2.line(frame,(cols-10,rows-10),(cols-10,rows-60),(0,100,250),2)
        cv2.line(frame,(cols-10,rows-10),(cols-60,rows-10),(0,100,250),2)

        cv2.line(frame,(int(cols/2),int(rows/2)-50),(int(cols/2),int(rows/2)+50),(0,0,250),1)
        cv2.line(frame,(int(cols/2)-50,int(rows/2)),(int(cols/2)+50,int(rows/2)),(0,0,250),1)

        cv2.circle(frame,(int(cols/2),int(rows/2)),20,(15,10,250), 2)

        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        lower_red=np.array([100,40,40])
        upper_red=np.array([255,255,150])
        
        mymask = cv2.inRange(hsv,lower_red,upper_red)
        res = cv2.bitwise_and(frame,frame,mask=mymask)
        
        cv2.imshow("Live video",frame)
        cv2.imshow("mask mode",mymask)
        cv2.imshow("res mode",res)
        if cv2.waitKey(1) and 0xFF ==ord('q'):
            break
    capture.release()
    quitFrame()

def saveVideo(dev):
    capture = cv2.VideoCapture(0)
    fourcc= cv2.VideoWriter_fourcc(*'XDIV')
    out = cv2.VideoWriter("C\\Users\\rootx\\Desktop\\myvideo.avi",20,(320,240))
    while(capture.isOpened()):
        ret,frame=capture.read()
        if ret==True:
            frame=cv2.flip(frame,0)
            out.wrtite(frame)
            cv2.imshow("outframe",frame)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
        else:
            break
    capture.release()
    out.release()
    quitFrame()

def padding(src):
    BLUE = [255,0,0]
    img1 = cv2.imread(src)
    replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
    reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
    reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
    wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
    constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)
    plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
    plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
    plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
    plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
    plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
    plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
    plt.show()

def sumImage(src1,src2,alpha=0):
    img1=cv2.imread(src1)
    img2 =cv2.imread(src2)
    dest = cv2.addWeighted(img1[0:300,0:300],0.5,img2[0:300,0:300],0.5,alpha)
    cv2.imshow("out",dest)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def threshMode(src,mode):
    img1=cv2.imread(src)
    ret,thresh_bin = cv2.threshold(img1,125,255,cv2.THRESH_BINARY)
    ret,thresh_bin_inv = cv2.threshold(img1,125,255,cv2.THRESH_BINARY_INV)
    ret,thresh_trunc = cv2.threshold(img1,125,255,cv2.THRESH_TRUNC)
    ret,thresh_tozero = cv2.threshold(img1,125,255,cv2.THRESH_TOZERO)
    ret,thresh_tozero_inv = cv2.threshold(img1,125,255,cv2.THRESH_TOZERO_INV)

    outs= ["Original","binary","binary inv","trunch","tozero","tozero inv"]
    images = [img1,thresh_bin,thresh_bin_inv,thresh_trunc,thresh_tozero,thresh_tozero_inv]

    if(mode==0):
        for i in range(6):
            cv2.imshow(outs[i],images[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif(mode==1):
        for i in range(6):
            #subplot(satır,sutun,deger)
            plt.subplot(2,3,i+1),plt.imshow(images[i])
            plt.title(outs[i])
            plt.xticks([]),plt.yticks([])
        plt.show()
    else:
        cv2.imshow("original",img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
def adaptiveThreshMode(src,mode):
    img1=cv2.imread(src,0)
    img_blur = cv2.medianBlur(img1,5)
    
    ret,thresh_binary = cv2.threshold(img_blur,127,255,cv2.THRESH_BINARY)
    adaptive_mean = cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    adaptive_gaussian = cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    outs= ["Original","binary","Mean C","Gaussian C"]
    images = [img_blur,thresh_binary,adaptive_mean,adaptive_gaussian]

    if(mode==0):
        for i in range(4):
            cv2.imshow(outs[i],images[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif(mode==1):
        for i in range(4):
            #subplot(satır,sutun,deger)
            plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
            plt.title(outs[i])
            plt.xticks([]),plt.yticks([])
        plt.show()
    else:
        cv2.imshow("original",img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def otsuThresh(src):
    img = cv2.imread(src,0)
    ret1,th1 = cv2.threshold(img,125,255,cv2.THRESH_BINARY)
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    images = [img,0,th1,
              img,0,th2,
              blur,0,th3]
    titles =['Orjinal','Histogram','global thresholding',
             'Orjinal','Histogram','Otsus thresholding',
             'Orjinal filtered image','Histogram','Otsu thresholding']

    for i in range(3):
        #subplot(satır,sutun,deger)
        plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
        plt.title(titles[i*3]),plt.xticks([]),plt.yticks([])
        plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
        plt.title(titles[i*3+1]),plt.xticks([]),plt.yticks([])
        plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
        plt.title(titles[i*3+1]),plt.xticks([]),plt.yticks([])
    plt.show()

def geometricProccess(src,mode):
    img=cv2.imread(src)
    print(type(img))
    result = None
    title=''
    rows,cols,ch=img.shape
    if(mode==0 or mode=='scale'):
        resize = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
        title='Scale Mode'
        result=resize
    elif(mode==1 or mode=='translate'):
        #[1 0 tx, 0 1 ty]
        title='Translate Mode'
        Mat = np.float32([[1,0,100],[0,1,50]])
        dest = cv2.warpAffine(img,Mat,(cols+100,rows+100))
        result = dest
    elif(mode==2 or mode=='rotate'):
        #matrix = [cos& -sin&, sin& cos&]
        title='Rotate Mode'
        Mat = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        dest = cv2.warpAffine(img,Mat,(cols+100,rows+100))
        result = dest
    elif(mode==3 or mode=='affine'):
        title='Affine mode'
        point1 = np.float32([[50,50],[200,50],[50,200]])
        point2 = np.float32([[10,100],[200,50],[100,250]])
        Mat = cv2.getAffineTransform(point1,point2)
        dest = cv2.warpAffine(img,Mat,(cols+200,rows+200))
        result = dest
    elif(mode==4 or mode=='perspective'):
        title = 'Perspective Mode'
        points1 = np.float32([ [0,0],[250,0],[0,250],[250,250] ])
        points2 = np.float32([ [45,60],[360,80],[30,370],[375,380] ])
        Mat = cv2.getPerspectiveTransform(points1,points2)
        dest = cv2.warpPerspective(img,Mat,(cols+200,rows+200))
        result = dest
    cv2.imshow('Original',img)
    cv2.imshow(title,result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def imageFiltering(src,mode):
    img=cv2.imread(src)
    result=None
    if(mode==0):
        kernel=np.ones( (5,5), np.float32)/25
        result=cv2.filter2D(img,-1,kernel)
    elif(mode==1):
        blur = cv2.blur(img,(5,5) )
        result = blur
    elif(mode==2):
        blur = cv2.GaussianBlur(img,(5,5),0 )
        result = blur
    elif(mode==3):
        blur = cv2.medianBlur(img,5 )
        result = blur
    elif(mode==4):
        blur = cv2.bilateralFilter(img,10,80,80 )
        result = blur
    cv2.imshow('Original',img)
    cv2.imshow("filter",result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def morphologicalTransform(src,mode):
    img = cv2.imread(src)
    result=None
    title=""
    kernel = np.ones( (5,5),np.uint8)
    if(mode==0 or mode=="erosion"):
        title = "erosion output"
        erosion = cv2.erode(img,kernel,iterations=1)
        result = erosion
    elif(mode==1 or mode=="dilation"):
        title = "dilation output"
        dilate = cv2.dilate(img,kernel,iterations=1)
        result = dilate
    elif(mode==2 or mode=="opening"):
        title = "opening output"
        opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
        result = opening
    elif(mode==3 or mode=="closing"):
        title = "closing output"
        closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
        result = closing
    elif(mode==4 or mode=="morph_gradient"):
        title = "gradient output"
        grad = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
        result = grad
    elif(mode==5 or mode=="tophat"):
        title = "tophat output"
        top = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
        result = top
    elif(mode==6 or mode=="blackhat"):
        title = "blackhat output"
        black = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
        result = black
    cv2.imshow('Original',img)
    cv2.imshow(title,result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gradients(src):
    img=cv2.imread(src)
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelX = cv2.Sobel( img,cv2.CV_64F,1,0,ksize=5 )
    sobelY = cv2.Sobel( img,cv2.CV_64F,0,1,ksize=5 )

    plt.subplot(2,2,1),plt.imshow(img)
    plt.title('Original'),plt.xticks([]),plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian)
    plt.title('Laplacian'),plt.xticks([]),plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelX)
    plt.title('Sobel X'),plt.xticks([]),plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobelY)
    plt.title('Sobel Y'),plt.xticks([]),plt.yticks([])
    plt.show()

def cannyFilter(src):
    img=cv2.imread(src)
    canny = cv2.Canny(img,100,300)
    
    plt.subplot(2,1,1),plt.imshow(img)
    plt.title('Original'),plt.xticks([]),plt.yticks([])
    plt.subplot(2,1,2),plt.imshow(canny)
    plt.title('Canny'),plt.xticks([]),plt.yticks([])
    plt.show()

def pyramids(src):
    img=cv2.imread(src)
    lowwer = cv2.pyrDown(None)

    cv2.imshow("Original",img)
    cv2.imshow("pry",lowwer)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def foundContours(src):
    img = cv2.imread(src,0)
    ret,thresh = cv2.threshold(img,127,255,0)
    contours,hierarchy = cv2.findContours(thresh,1,2)

    cnt=contours[0]
    Mv = cv2.moments(cnt)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    epsilon = 0.1*(cv2.arcLength(cnt,True))
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    convex = cv2.convexHull(cnt)
    isCheck = cv2.isContourConvex(cnt)
    print("cnt:",cnt)
    print("moment: ",Mv)
    print("area:",area)
    print("perim:",perimeter)
    print("epsilon:",epsilon)
    print("approx:",approx)
    print("convex:",convex)
    print("is Convex:",isCheck)

    x,y,width,height = cv2.boundingRect(cnt)
    imgT2 = cv2.imread(src,0)
    img2 = cv2.rectangle(imgT2,(x,y),(x+width-80,y+height-100),(0,0,255),5)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    imgT3 = cv2.imread(src,0)
    img3 = cv2.drawContours( imgT3,[box],0,(0,255,0),2 )

    (x_1,y_1),radius = cv2.minEnclosingCircle(cnt)
    center = ( int(x_1),int(y_1) )
    radius = int(radius)
    imgT4 = cv2.imread(src,0)
    img4 = cv2.circle(imgT4,center,radius,(0,255,0),2)

    plt.subplot(2,2,1),plt.imshow(img)
    plt.title('Original'),plt.xticks([]),plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(img2)
    plt.title('rectangle'),plt.xticks([]),plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(img3)
    plt.title('drawContours'),plt.xticks([]),plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(img4)
    plt.title('Enclosing'),plt.xticks([]),plt.yticks([])
    plt.show()

def histogramMask(src):
    img=cv2.imread(src)

    mymask = np.zeros(img.shape[:2],np.uint8)
    mymask[100:300, 100:380] = 255
    masked_img = cv2.bitwise_and(img,img,mask = mymask)

    hist_full = cv2.calcHist([img],[0],None,[2556],[0,256])
    hist_mask =  cv2.calcHist([img],[0],mymask,[2556],[0,256])

    plt.subplot(221),plt.imshow(img,'gray')
    plt.subplot(222),plt.imshow(mymask,'gray')
    plt.subplot(223),plt.imshow(masked_img,'gray')
    plt.subplot(224),plt.plot(hist_full),plt.plot(hist_mask)
    plt.xlim([0,256])
    plt.show()
    
    
#histogramMask(src)
#foundContours(src)   
#pyramids(src)
#cannyFilter(src)
#gradients(src)
#morphologicalTransform(src,6)    
#imageFiltering(src,4)
#geometricProccess(src,1)
#otsuThresh(src)
#adaptiveThreshMode(src,1)
#threshMode(src,1)
#openCamera(0)
#saveVideo(0)
#openCamera("F:\\belgesel\\myvideo.mp4")
#showPlot(src)
#readImage("C:\\Users\\rootx\\Desktop\\watt.JPG",0)
#writeImage("C:\\Users\\rootx\\Desktop\\watt.JPG",0,"C:\\Users\\rootx\\Desktop\\watt2.JPG")

