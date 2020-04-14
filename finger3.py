#------------------------------------------------------------
#------------------------------------------------------------
import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise

# aarkaplan global degisken
bg = None

#--------------------------------------------------
# Arka plandaki ortalama.
#--------------------------------------------------
def run_avg(image, accumWeight):
    global bg
    # arkaplan baslatıcı
    if bg is None:
        bg = image.copy().astype("float")
        return

    # ağırlıklı ortalama hesspla ve arka planı güncelle
    cv2.accumulateWeighted(image, bg, accumWeight)

#----------------------------------------------------
# Görüntüdeki el icin ozel bir bölüm olusturan metot.
#----------------------------------------------------
def segment(image, threshold=25):
    global bg
    # arkaplan ve geçerli frame arasındaki mutlak farkı bul.
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # diff görüntüsünü eşikleyerek ön planı elde edelim
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # eşikli görüntüdeki contour lari elde edelim
    ( cnts, hierarchy) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # contour tespit edilmese bir sey yapmma
    if len(cnts) == 0:
        return
    else:
        # Countour alanına göre,mevcud max contouru al.
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#--------------------------------------------------------------
# Ozel El bölgedesinde parmak sayısını sayan metot.
#--------------------------------------------------------------
def count(thresholded, segmented):
    # parcali el bölgesinin convex-hulldegeri bul.
    chull = cv2.convexHull(segmented)
    print(chull)
    # convex hull de en uç noktaları bul.
    # ust,alt,sol,sag
    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

    # avucunun merkezini hesapla.
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    # avuç içi merkezi ile(X-Xo)^2+(Y-Y0)^2
    # dışbükey gövdenin en uç noktaları arasındaki
    # maksimum öklid mesafesini bulun
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    # elde edilen maksimum öklid mesafesinin yuzde 80i ile dairenin yarıçapını hesapla.
    radius = int(0.8 * maximum_distance)

    # dairenin çevresini hesapla
    circumference = (2 * np.pi * radius)

    # avuç içi ve parmakları olan dairesel ilgi bölgesini çıkar.
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
	
    # ROI daire ciz
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    # maske olarak dairesel ROI kullanarak eşikli el arasında -bitwiseAND- al.
    # Burada eşikli el görüntüsünde maske kullanılarak elde edilen kesimleri verir.
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # dairesel ROI'deki contour ları hesapla.
    (cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # parmak sayicisi
    count = 0

    # tespit edilen countour lar arasindaki dongu
    for c in cnts:
        # konturun sınırlayıcı kutusunu hesaplar
        (x, y, w, h) = cv2.boundingRect(c)

        # Parmak sayisi
        # 1. Bottom Alani
        # 2. Contour boyunca nokta sayısı, dairesel ROI'nin
        # çevresinin yuzde 25'ini geçmez.
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count

#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":
    
    accumWeight = 0.5
    # web kamerası referansi
    camera = cv2.VideoCapture(0)
    # Ozel bolme koordinatlarimiz
    top, right, bottom, left = 10, 350, 225, 590
    # balangic kare sayisı
    num_frames = 0
    # ayarlama gostergesi balangıcta etkin degil
    calibrated = False

    # Dongu
    while(True):
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=700)

        # flip goruntudeki tersligi duzeltir(Ayna Görunumu)
        frame = cv2.flip(frame, 1)

        clone = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # arka planı elde etmek için,
        # ağırlıklı ortalama modelimizin kalibre edilmesi için bir eşiğe ulaşılana kadar
        # aramaya devam etsin.
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("[STATUS] lutfen bekleyiniz! ayarlaniyor......")
            elif num_frames == 29:
                print("[STATUS] ayarlama başarili")
        else:
            hand = segment(gray)
            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                fingers = count(thresholded, segmented)
                cv2.putText(clone, str(fingers)+" parmak", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow("Thesholded", thresholded)


        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        num_frames += 1

        cv2.imshow("Video Feed", clone)
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break


camera.release()
cv2.destroyAllWindows()
