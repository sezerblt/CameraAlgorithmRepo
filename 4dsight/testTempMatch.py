import cv2
from matplotlib import pyplot as plt
import argparse
import glob
"""
import argparse
import glob

a = argparse.ArgumentParser()
a.add_argument("-t","--template", required=True,help="path of template figure")
a.add_argument("-i","--image",    required=True,help="path of image figure")
a.add_argument("-m","--match_type",type=str,default="TM_CCOEF",help="select matching type")

args = vars(a.parse_args())


template = cv2.imread(args["template"])
image = cv2.imread(args["image"])
match_type  = glob.glob("cv2."+args["match_type"])
"""


a = argparse.ArgumentParser()
a.add_argument("-t","--template", required=True,help="path of template file")
a.add_argument("-i","--image",    required=True,help="path of image file")
#a.add_argument("-m","--match_type",type=int,default=cv2.TM_CCOEF",help="select matching type")

args = vars(a.parse_args())
X=None
def methodsMatch(method):
    methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

    if method==1:
        return methods[0]
    elif method==2:
        return  methods[1]
    elif method==3:
        return  methods[2]
    elif method==4:
        return  methods[3]
    elif method==5:
        return  methods[4]
    elif method==6:
        return  methods[5]

def detectTemplate(image,template,T_W,T_H,match_type):
    """
    @image: input image
    @template: our template file
    @T_W: weight of template file
    @T_H: height of template file
    """
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray,template,match_type)

    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(result)
    print("min_val: ",min_val)
    print("max_val: ",max_val)
    print("min_loc: ",min_loc[0],":",min_loc[1])
    print("max_loc: ",max_loc[0],":",max_loc[1])

    top_left=min_loc
    bottom_right=(top_left[0]+T_W,top_left[1]+T_H)
    return (max_val,top_left,bottom_right)

if __name__ == '__main__':
    print("""
    |--------------------------------------------------------------------------
    |  Wellcome Mini Template Matching Test Program : version V.0.0.1         
    |--------------------------------------------------------------------------
    |1.write for Linux/OS;
    |>>python testTempMatch.py --image "IMAGE_PATH" --template "TEMPLATE_PATH";
    |   or
    |>>python testTempMatch.py -i      "IMAGE_PATH" -t         "TEMPLATE_PATH";
    |
    |--------------------------------------------------------------------------
    |1.write for Windows;
    |>>py testTempMatch.py --image "IMAGE_PATH" --template "TEMPLATE_PATH";
    |    or
    |>>py testTempMatch.py    -i "IMAGE_PATH"   -t         "TEMPLATE_PATH";
    |
    |#-------------------------------------------------------------------------
    |2.After;
    |choice template match methods between at (1-6):
    |       1.TM_CCOEFF
    |       2.TM_CCOEFF_NORMED
    |       3.TM_CCORR
    |       4.TM_CCORR_NORMED
    |       5.TM_SQDIFF
    |       6.TM_SQDIFF_NORMED
    |for example;
    |>>?: 1
    |(choice: TM_CCOEF) PRESS ENTER KEY
    |     >>>>>Result<<<<<<
    |--------------------------------------------------------------------------
    """)
    
    #image_file = "StarMap.png"
    #template_file = "Small_area_rotated.png"
    #image = cv2.imread(image_file)
    #template = cv2.imread(template_file,0)
    
    
    image = cv2.imread(args["image"])
    template = cv2.imread(args["template"],0)

    print(""" Select template match methods :
            1.TM_CCOEFF
            2.TM_CCOEFF_NORMED
            3.TM_CCORR
            4.TM_CCORR_NORMED
            5.TM_SQDIFF
            6.TM_SQDIFF_NORMED
          """)
    method =int(input("?:"))
    match_type= methodsMatch(method)
    

    T_H,T_W = template.shape[:2]
    result_detect = detectTemplate(image,template,T_W,T_H,match_type)
    print("result: ",result_detect)

    cv2.rectangle(image,*result_detect[1:],(255,0,0),3)
    cv2.putText(image,"FIX",(result_detect[1][0],result_detect[1][1]),cv2.FONT_HERSHEY_PLAIN,3,(50,250,50))
    

    image=cv2.resize(image,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
    template=cv2.resize(template,None,fx=2,fy=2,interpolation=cv2.INTER_AREA)
    #result=cv2.resize(result,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)

    cv2.imshow("Input",image)
    cv2.imshow("template",template)
    #cv2.imshow("Matcing",result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
"""
#Test Image
image_file = "StarMap.png"
template_file = "Small_area_rotated.png"
match_type= cv2.TM_SQDIFF

image = cv2.imread(image_file)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
template = cv2.imread(template_file,0)

T_H,T_W = template.shape[:2]

result = cv2.matchTemplate(gray,template,match_type)

min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(result)
print("min_val: ",min_val)
print("max_val: ",max_val)
print("min_loc: ",min_loc[0],":",min_loc[1])
print("max_loc: ",max_loc[0],":",max_loc[1])

top_left=min_loc
bottom_right=(top_left[0]+T_W,top_left[1]+T_H)

cv2.rectangle(image,top_left,bottom_right,(255,0,0),3)
cv2.putText(image,"FIX",(top_left[0]-50,top_left[1]-20),cv2.FONT_HERSHEY_PLAIN,3,(50,250,50))
cv2.rectangle(result,top_left,bottom_right,(0,0,0),5)

image=cv2.resize(image,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
template=cv2.resize(template,None,fx=2,fy=2,interpolation=cv2.INTER_AREA)
result=cv2.resize(result,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)


cv2.imshow("Input",image)
cv2.imshow("template",template)
cv2.imshow("Matcing",result)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
