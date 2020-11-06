import numpy as np
import argparse
import imutils
import glob
import cv2

a = argparse.ArgumentParser()
a.add_argument("-t","--template", required=True,help="path of template figure")
a.add_argument("-i","--images",    required=True,help="path of image figure")
a.add_argument("-v","--visualize",help="visualize each iteration")

args = vars(a.parse_args())


class TemplateMatch:
    def __init__(self,methodType):
        self.image = cv2.imread(args["image"]);
        self.gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);
        self.template = cv2.imread(args["template"],0)
        self.method = methodType;
        
        self.template_width,self.template_height = self.template.shape[::-1]

        self.matching()

    def matching(self):
        self.match_template = cv2.matchTemplate(self.gray,self.template,self.method)

        position = np.where(self.match_template >= 0.7)

        for  px_py in zip(*position[::-1]):
            cv2.rectangle(self.image,px_py,(px_py[0]+self.template_width,px_py[1]+self.template_height),(80,100,255),2)
        cv2.imshow("template Result",self.image)
        cv2.waitKey(0)
#----------------------------------------------------------------------------------------
class MultiTemplateMatch:
    def __init__(self):
        self.template = cv2.imread(args["template"])
        self.gray_template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
        self.canny_template = cv2.Canny(gray_template,50,255)
        self.temp_H, self.temp_W = self.canny_template.shape[:2]

        self.showTemplate()
        self.multiMatching()

    def showTemplate(self):
        cv2.imshow("canny figure",self.canny_template)
        cv2.waitKey(0)

    def multiMatching(self):

        for glb in glob.glob(args["images"]+"/*.png" or "/*.jpg"):
            self.image = cv2.imread(glb)
            self.gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            self.isTemplateFound = None

            for scale in np.linspace(0.1,1.0,50)[::-1]:
                RESIZE = imutils.resize(self.gray_image,width = int(self.gray_image.shape[1]*scale))
                r = self.gray_image.shape[1]/float(RESIZE.shape[1])

                if(RESIZE.shape[0]<self.temp_H or RESIZE.shape[1]<self.temp_W):
                    break

                edged = cv2.Canny(RESIZE,50,250)

                result == cv2.matchTemplate(edged,self.template,cv2.TM_CCOEFF)

                minVal,maxVal,minLoc,maxLoc = cv2.minMaxLoc(result)

                if(args.get("visualize",False)):
                    visualize = np.dstack([edged,edged,edged])
                    cv2.rectangle(visualize,maxLoc[0],maxLoc[1],(maxLoc[0]+self.temp_W,maxLoc[1]+self.temp_H),(0,255,0),1)
                    cv2.imshow("visualize figure",visualize)
                    cv2.waitKey(0)

                if (self.isTemplateFound is None) or (maxVal > self.isTemplateFound[0]):
                    self.isTemplateFound = (maxVal,maxLoc,r)
                    
            (maxval,maxLoc,r)=self.isTemplateFound
            (x0,y0) = ( int(maxLoc[0]*r),int(maxLoc[1]*r))
            (xW,yH) = ( int((maxLoc[0]+self.temp_W)*r),int((maxLoc[1]+self.temp_H)*r))

            cv2.rectangle(self.image,(x0,y0),(xW,yH),(250,0,0),2)
            cv2.imshow("Input Image",self.image)
            cv2.waitKey(0)
                    
    
if __name__ == '__main__':
    method=cv2.TM_SQDIFF
    tm =TemplateMatch(method)
        
        






































