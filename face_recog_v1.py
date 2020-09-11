import cv2 as cv
import argparse
import numpy

class FaceRecognition():
    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
    parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
    args = parser.parse_args()
    face_cascade_name = args.face_cascade
    eyes_cascade_name = args.eyes_cascade
    
    face_cascade = cv.CascadeClassifier()
    eyes_cascade = cv.CascadeClassifier()
    
    def __init__(self):
        
        if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
            print('--(!)Error loading face cascade')
            exit(0)
        if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
            print('--(!)Error loading eyes cascade')
            exit(0)
            
        if not cap.isOpened:
            print('--(!)Error opening video capture')
            exit(0)
        
        self.camera_device = args.camera
        self.capture = cv.VideoCapture(self.camera_device)
            
        while True:
            self.ret, self.frame = self.capture.read()
            if frame is None:
                print('--(!) No captured frame -- Break!')
                break
            detectAndDisplay(frame)
            if cv.waitKey(10) == 27:
                break

    def detectAndDisplay(frame):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)
        #-- Face
        faces = face_cascade.detectMultiScale(frame_gray)
        for (x,y,w,h) in faces:
            """
                x = x_faceArea_point_width
                y = y_faceArea_point_height
                w = lenght_weight
                h = lenght_height
            """
            center = (x + w//2, y + h//2)
            frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
            faceROI = frame_gray[y:y+h,x:x+w]
            #--Eyes 
            eyes = eyes_cascade.detectMultiScale(faceROI)
            for (x2,y2,w2,h2) in eyes:
                """
                    x2 = x_eyeArea_point_width
                    y2 = y_faceArea_point_height
                    w2 = lenght_weight_of_eyeArea
                    h2 = lenght_height_of_eyearea
                """
                eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                radius = int(round((w2 + h2)*0.25))
                frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
        cv.imshow('Yuz Tanımlama', frame)

f=FaceRecognition()
f
"""
#test
def detectAndDisplay(frame):
    font = cv.FONT_HERSHEY_SIMPLEX
    frame=cv.flip(frame,1)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.equalizeHist(frame)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame)
    #print(faces)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        #frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (20, 250, 20), 4)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faceROI = frame_gray[y:y+h,x:x+w]
        cv.line(frame,(x + w//2, y + h//2+20),(x + w//2,y+h),(0,200,200),2)
        cv.putText(frame,'Yuz',(x, y-10), font, 1,(60,25,5),1,cv.LINE_AA)
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        #print(eyes)
        if not numpy.any(eyes):
            cv.putText(frame,'Goz Tespit Edilmedi',(x, y + h), font, 1,(25,25,250),1)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.2))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 2)
            cv.putText(frame,'Gozler',(x + x2 + w2//2, y + y2 + h2//2-10), font, 1,(25,25,25),1)
            cv.line(frame,(x + w//2, y + h//2+20),eye_center,(0,200,200),2)
            cv.line(frame,(x + w//2, y + y2 + h2//2),eye_center,(200,200,0),2)
    cv.imshow('Capture - Face detection', frame)
    

face_cascade = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
eyes_cascade = cv.CascadeClassifier("haarcascades/haarcascade_eye.xml")
cap = cv.VideoCapture(0)
if not cap.isOpened:
    print('--(!)Hata Kamera açılmadı')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) Görüntü algılanamadı')
        break
    detectAndDisplay(frame)
    if cv.waitKey(10) == 27 or cv.waitKey(10) == 0xFF:
        break   

cap.release()
cv.destroyAllWindows()

while True:
    #Capture frame-by-frame
    ret, frame = video_capture.read()
    frame=cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('FaceDetection', frame)
    #ESC Pressed
    key = cv2.waitKey(10) 
    if key == 27: 
ı        break
video_capture.release()
cv2.destroyAllWindows()
"""
