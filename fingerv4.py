# Imports
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
# Open Camera
'''capture = cv2.VideoCapture('http://192.168.43.131:4747/video')'''
capture = cv2.VideoCapture(0)
while capture.isOpened():

    # kameradan gelen görüntü verisini al
    ret, frame = capture.read()
    frame=cv2.flip(frame,1)

    # işlem için bir alan ayarla/belirle
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
    crop_image = frame[100:300, 100:300]
    
    # Gaussian ile görüntüyü temizle
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    # renk uzayını BGR -> HSV olarak ayarla
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # binary goruntu için maske
    mask2 = cv2.inRange(hsv, np.array([0, 48, 80]), np.array([20, 255, 255]))
    #cv2.imshow("mask2", mask2)
    # 5x5 lik birrlik matrisi -morfolojik işlemler icin-
    kernel = np.ones((3, 3))

    # erode ve dilate işlemi ile görüntü iyilestirme
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # Gaussian Blur ve Threshold filterleme sonrasında esik degeri belirle
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    #cv2.imshow("filtered", filtered)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    # Threshold imag goster
    cv2.imshow("Threshold", thresh)

    # esik degerdeki göruntunun kenarları tespit et
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    try:
       
        # İslem bolgesindeki kenarları tespit et
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        
        # hatlardan dortgen olustur
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # Convex hull metodu dış bukey keanarlrı ayarla
        hull = cv2.convexHull(contour)

        # Hattı ciz
        drawing = np.zeros(crop_image.shape, np.uint8)
        
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        #parmaklar icin sayac olustur
        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            # 90 dereceden buyukacılar icin daire cizimi
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image, far, 1, [0, 0, 255], -1)

            cv2.line(crop_image, start, end, [0, 255, 0], 2)

        # Ekranda parmak sayısını yazdır
        if count_defects == 0:
            cv2.putText(frame, "Bir", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
        elif count_defects == 1:
            cv2.putText(frame, "İki", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        elif count_defects == 2:
            cv2.putText(frame, "Uc",  (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        elif count_defects == 3:
            cv2.putText(frame, "Dort", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        elif count_defects == 4:
            cv2.putText(frame, "Bes",  (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        else:
            cv2.imshow('mask',mask)
            cv2.imshow('frame',frame)
            pass
    except:
        pass

    # istenilenresim
    cv2.imshow("frame output", frame)
    
    #all_image = np.hstack((drawing, crop_image))
    ##cv2.imshow('Contours', all_image)

    # cıkıs icin 'q' tusuna bas
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
