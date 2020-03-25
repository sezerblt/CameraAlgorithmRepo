import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to the image file")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
for angle in np.arange(0, 360, 15):
    rotated = imutils.rotate(image, angle)
    cv2.imshow("Rotated (Problematic)", rotated)
    cv2.waitKey(0)

for angle in np.arange(0, 360, 15):
    rotated = imutils.rotate_bound(image, angle)
    cv2.imshow("Rotated (Correct)", rotated)
    cv2.waitKey(0)
"""
mat=[[a,b,1-a]
"""
def rotate_bound(image, angle):
    #resmin boyutlarını al
    # merkez
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    
    #donus matrisini al
    #saat yönünde döndür(sin,cos)
    #(matrisin döndrme birlesenleri)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    # goruntunun boyutlarını hesapla
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    # donus matrisini ayarla
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    
    #donus islemimi yap
    return cv2.warpAffine(image, M, (nW, nH))
