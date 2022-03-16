import cv2
import numpy as np

class HarrisCornerImage(object):
    def __init__(self,image):
        self._image = cv2.imread(image)

    @property
    def image(self):
        return self._image

    def findCorner(self):
        gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 10, 23, 0.04)
        self._image[dst>0.01 * dst.max()] = [0, 0, 255]


class SIFTImage(HarrisCornerImage):
    def __init__(self,image):
        self._image = cv2.imread(image)
        self._gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        self._sift = cv2.xfeatures2d.SIFT_create()
        

    def DoG(self):
        keypoints, descriptor = self._sift.detectAndCompute(self._gray,None)
        self._image = cv2.drawKeypoints(
            image=self._image, outImage=self._image, keypoints =keypoints
        )




    