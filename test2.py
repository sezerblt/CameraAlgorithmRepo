import cv2
import numpy as np
from matplotlib import pyplot as plt
#-------------------------------------------
src="C:\\Users\\rootx\\Desktop\\car.JPG"
src2="C:\\Users\\rootx\\Desktop\\face1.JPG"
src3="C:\\Users\\rootx\\Desktop\\face2.JPG"
src4="C:\\Users\\rootx\\Desktop\\watt.JPG"
simple="C:\\Users\\rootx\\Desktop\\simple.png"
#--------------------------------------------

def plot2d(src):
    img = cv2.imread(src2)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 50, 0, 256] )
    plt.imshow(hist)
    plt.show()


def histback(src):
    roi = cv2.imread(src)
    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    
    target = cv2.imread(src)
    hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
    
    roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )

    cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
    dest = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(dest,-1,disc,dest)

    ret,thresh = cv2.threshold(dest,50,255,0)
    thresh = cv2.merge( (thresh,thresh,thresh) )
    res = cv2.bitwise_and(target,thresh)

    res = np.vstack( (target,thresh,res) )
    cv2.imwrite('dat.jpg',res)

def fourierTransform(src):
    # x(t) = A*sin(2*pi*f*t)   :  Discrete fourier Transform(DTF)
    img = cv2.imread(src,0)

    dift = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dift)
    #magnitude_spectrum1 =     1*(np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])))
    #magnitude_spectrum100 = 10000*(np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])))
    rows, cols = img.shape
    crow,ccol = int(rows/2) , int(cols/2)

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-50:crow+50, ccol-50:ccol+50] = 1
    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    plt.subplot(121),plt.imshow(img)
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img_back)
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])    
    plt.show()
    
def templateMatching(src,src2):
    img = cv2.imread(src,0)
    img2 = img.copy()
    template = cv2.imread(src2,0)
    width,height = template.shape[::-1]

    methods = ['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR','cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']

    for m in methods:
        img = img2.copy()
        method = eval(m)

        result = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = ( top_left[0]+width,top_left[1]+height)
        cv2.rectangle(img,top_left,bottom_right,255,2)

        plt.subplot(121),plt.imshow(result,cmap='gray')
        plt.title('Match Template'),plt.xticks([]),plt.yticks([])
        plt.subplot(122),plt.imshow(result,cmap='gray')
        plt.title('Detected Match'),plt.xticks([]),plt.yticks([])
        plt.suptitle(m)
        plt.show()
        
def tempMulti(src):
    imgRGB = cv2.imread(src)
    imgGRAY = cv2.cvtColor(imgRGB,cv2.COLOR_BGR2GRAY)
    template = cv2.imread(src,0)
    width,height = template.shape[::-1]
    res = cv2.matchTemplate(imgGRAY,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0,5
    loc = np.where(res >= threshold)
    for pnt in zip(*loc[::-1]):
        cv2.rectangle(imgRGB,pnt,(pnt[0]+width,pnt[1]+height),(0,0,255),2)
    cv2.imwrite('result.JPG',imgRGB)
        
def tempMatch(src,src2):
    img = cv2.imread(src,0)
    img2 = img.copy()
    template = cv2.imread(src2,0)
    w,h = template.shape[::-1]
    methods =['cv2.TM_CCOEFF','cv2TM_CCOEFF_NORMED',
              'cv2.TM_CCORR','cv2.TM_CCORR_NORMED',
              'cv2.TM_SQDIFF','cv2.SQDIFF_NORMED']
    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        res = cv2.matchTemplate(img,template,method)
        min_val,max_val,min_loc,max_loc =cv2.minMaxLoc(res)

        if method in [cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (img,top_left[0]+w,top_left[1]+h)
        
        cv2.rectangle(img,top_left,bottom_right,255,2)

        plt.subplot(121),plt.imshow(res,cmap='gray')
        plt.title('Matching Result'),plt.xticks([]),plt.yticks([])
        plt.subplot(122),plt.imshow(img,cmap='gray')
        plt.title('Detected Point'),plt.xticks([]),plt.yticks([])
        plt.subtitle(meth)


        plt.show()

tempMatch(src,src2)
#sayfa136 template matching
