import cv2 as cv
import numpy as np

SZ=20
bin_n = 16 # içerik sayııs


affine_flags = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR

## bir rakam görüntüsü alan ve eçarıklıgı duzelten fonksiyon
def deskew(img):
    moment = cv.moments(img)
    if abs(moment['mu02']) < 1e-2:
        return img.copy()
    skew = moment['mu11']/moment['mu02']
    Matrix = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv.warpAffine(img,Matrix,(SZ, SZ),flags=affine_flags)
    return img
## [deskew]

## [hog]
def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    
    mag, ang = cv.cartToPolar(gx, gy)

    bins = np.int32(bin_n*ang/(2*np.pi))    # bin_degerleri(0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]

    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     #64 bit vector histogram
    return hist


img = cv.imread('digits.png',0)
if img is None:
    raise Exception("digits.png goruntu dosyası bulunamadi !")

#goruntu ayır ve hucrelere bol
#C[1,1],C[1,2],   ...  ,C[1:100]
#C[2,1],C[2,2],   ...  ,C[2:100]
#   ...           ...    ...
#C[50,1],C[50,2], ...  ,C[50,100]
cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]

#test ve egitim hucreleri listele
train_cells = [ i[:50] for i in cells ]
test_cells = [ i[50:] for i in cells]

######     egitim baslat ve listeler olustur    ########################

deskewed = [list(map(deskew,row)) for row in train_cells]
hogdata = [list(map(hog,row)) for row in deskewed]
trainData = np.float32(hogdata).reshape(-1,64)
responses = np.repeat(np.arange(10),250)[:,np.newaxis]

#svm vector olustur
svm = cv.ml.SVM_create()
#svm vector 'e dogrusal filtre uygula
svm.setKernel(cv.ml.SVM_LINEAR)
#svm vector turunu SVC yap
svm.setType(cv.ml.SVM_C_SVC)
#svm icin Sabit degeri ve gamma olcusu ata
svm.setC(2.67)
svm.setGamma(5.383)
#svm vector datayı egit
svm.train(trainData, cv.ml.ROW_SAMPLE, responses)
#svm kaydet
svm.save('svm_data.dat')

######     testi baslat      ########################

#hucrelerdeki duzeltilen veriyi listele
deskewed = [list(map(deskew,row)) for row in test_cells]
#hog listenine ayarla ve liste olustur
hogdata = [list(map(hog,row)) for row in deskewed]
#test datası lustur
testData = np.float32(hogdata).reshape(-1,bin_n*4)
#predict tahmini fonksiyona gonder ve tahmini deger dondur
result = svm.predict(testData)[1]

#######   Correct islemi   ########################
mask = result==responses
correct = np.count_nonzero(mask)
print(correct*100.0/result.size)
cv.imshow("",img)
