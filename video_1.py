import numpy as np
import cv2 as cv
import argparse

def test():
    drawing = False 
    mode = True 
    ix,iy = -1,-1
    # mouse callback fopnksiyonları
    def draw_circle(event,x,y,flags,param):
        global ix,iy,drawing,mode
        red=(255,0,0)
        green=(0,255,0)
        blue=(0,0,255)
        black=(0,0,0)
        display=(250,250,250)
        
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = x,y
            print('--mouse--SOl TUS--Basili')
        elif event == cv.EVENT_MOUSEMOVE:
            if drawing == True:
                if mode == True:
                    cv.rectangle(img,(ix,iy),(x,y),green,-1)
                    print('--mouse--AKTIF')
                    print('-DORTGEN-')
                else:
                    cv.circle(img,(x,y),5,(0,0,255),-1)
                    print('--mouse--AKTIF')
                    print('-CEMBER-')

        elif event == cv.EVENT_LBUTTONUP:
            drawing = False
            print('--mouse--SOl TUS--Serbest')
            if mode == True:
                cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv.circle(img,(x,y),5,(0,0,255),-1)
            
    img = np.ones((512,512,3), np.uint8)
    cv.namedWindow('image')
    cv.setMouseCallback('image',draw_circle)
    while(1):
        cv.imshow('image',img)
        k = cv.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 27:
            break
    cv.destroyAllWindows()


def opencamera():
    img='C:\\Users\\rootx\\Downloads\\logos\\mic.png'
    print(img)
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("kamera acilmadi")
        exit()
    count=0
    while True:
        count+=1
        print('goruntu sayısı(t):',count)
        # her bir görüntü karesi al
        ret, frame = cap.read()
        h,w,ch=frame.shape
        # eger dönüs değeri yoksa - retrieval
        if not ret:
            print("video akisi saglanamiyor.Cikis yapiliyor...")
            break
        #renkli kare gri formata dönusturur.
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        """
        if cv.waitKey(0)==ord('l'):
            cv.line(gray,(0,0),(500,500),(255,0,0),5)
        if cv.waitKey(0)==ord('r'):
            cv.rectangle(gray,(0,0),(500,500),(255,0,0),5)
        if cv.waitKey(0)==ord('c'):
            cv.circle(gray,(0,0), 10,(0,200,0),1)
        """
        if count%2 == 0:
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(gray,'GRAY IMAGE TEXT',(int(w/5),int(h/3)),font,2,(0,0,0),1,cv.LINE_AA)
        # kare ekrana verilsin
        frame = cv.copyMakeBorder(frame,10,10,10,10,cv.BORDER_CONSTANT,value=[255,0,0])
        hsv= cv.copyMakeBorder(hsv,10,10,10,10,cv.BORDER_CONSTANT,value=[0,0,255])
        cv.imshow('frame', gray)
        cv.imshow('reflect',frame)
        cv.imshow('red_constant',hsv)
        #cv.imshow('320x320',frame[0:320,0:320])
        if cv.waitKey(1) == ord('q'):
            break
    # islem tamamlandiginda, görüntü serbest bırakılır
    cap.release()
    cv.destroyAllWindows()
    
opencamera()

#-------------------------
def playVideo(source):
    cap = cv.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("video akisi saglanamiyor.Cikis yapiliyor..")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

def saveVideo():
    cap = cv.VideoCapture(0)
    # VideoWriter nesnesi olustur ve codec tanimla
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    #
    out = cv.VideoWriter('output.avi', fourcc, 20.0, (640,480))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("video akisi saglanamiyor.Cikis yapiliyor...")
            break
        frame = cv.flip(frame, 1)
        # flip için obje yazsın
        out.write(frame)
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv.destroyAllWindows()

