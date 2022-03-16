from importlib.resources import path
import os,sys,pathlib,cv2

"""
def getImageList(index=0):
    str_filepath=os.path.join(pathlib.Path(__file__).resolve().parent,"bmw10_ims")
    path_file=pathlib.Path(str_filepath)
    print("path:",type(str_filepath))
    print("pdata:",type(path_file))
    datalist =list( path_file.glob("*/*.jpg") )
    img=cv2.imread(str(datalist[index]))
    
    cv2.imshow("in",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

class ImageUtil(object):
    def __init__(self,imgFiles):
        self.file=imgFiles
        pathfile=path.Path(os.path.join(pathlib.Path(__file__).resolve().parent,imgFiles))
        self._dataList=list( pathfile.glob("*/*.jpg") )
        self._image=None

    def fileInfo(cls):
        print(os.path.dirname(cls.file))

    @property
    def dataList(self):
        return self._dataList

    @property
    def image(self):
        if self._image is None:
            return self.getImageFromIndex(0)
        else:
            return self._image

    def getImageFromIndex(self,index):
        img_list=self.dataList
        img=cv2.imread(str(img_list[0]))
        return img
