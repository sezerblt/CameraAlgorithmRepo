from ast import arg
import cv2
import numpy as np


class Show(object):
    def __init__(self,frame):
        self.run(frame)
    def run(self,frame):
        cv2.imshow("in",frame)


    

class Tracking(object):
    def __init__(self,value=0):
        self.val=value
        self.run()

    def run(self):
        camera=cv2.VideoCapture(self.val)
        es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,3))
        kernel = np.ones((5,5),np.uint8)
        background = None
        while (True):
            ret, frame = camera.read()
            if background is None:
                background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                background = cv2.GaussianBlur(background, (21, 21), 0)
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
            diff = cv2.absdiff(background, gray_frame)
            diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            diff = cv2.dilate(diff, es, iterations = 2)
            cnts, hierarchy = cv2.findContours(
                diff.copy(),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            for c in cnts:
                if cv2.contourArea(c) < 1500:
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0), 2
                )
            
            cv2.imshow("contours", frame)
            cv2.imshow("dif", diff)
            if cv2.waitKey(10) & 0xff == ord("q"):
                break
        cv2.destroyAllWindows()
        camera.release()

class TrackingMOG(object):
    def __init__(self,value):
        self.val=value
        self.run()

    def run(self):
        bs = cv2.createBackgroundSubtractorKNN(detectShadows = True)
        camera = cv2.VideoCapture(self.val)
        struElement=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        while(True):
            ret, frame = camera.read()
            fgmask = bs.apply(frame)

            th = cv2.threshold(fgmask.copy(), 200, 255,cv2.THRESH_BINARY)[1]
            dilated = cv2.dilate(th,struElement,iterations = 2)

            contours, hier = cv2.findContours(dilated,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) > 1600:
                    (x,y,w,h) = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 250), 2)
            cv2.imshow("mog", fgmask)
            cv2.imshow("thresh", th)
            cv2.imshow("detection", frame)
            if cv2.waitKey(10) & 0xff == ord("q"):
                break
        cv2.destroyAllWindows()
        camera.release()

class MeanSHIFT(object):
    
    @classmethod
    def test(cls):
        cap = cv2.VideoCapture(0)

        ret,frame = cap.read()
        r,h,c,w = 10, 200, 10, 200
        track_window = (c,r,w,h)
        roi = frame[r:r+h, c:c+w]

        hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    
        lower_array=np.array( (100.,  30.,32.) )
        upper_array=np.array( (180.,120.,255.) )
        mask = cv2.inRange(hsv_roi, lower_array,upper_array)
        cv2.imshow("mask", mask)

        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,1 )

        while True:
            ret ,frame = cap.read()
            if ret == True:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

                ret, track_window = cv2.meanShift(dst, track_window,term_crit)
                x,y,w,h = track_window
                img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)

                cv2.imshow("frame", frame)
                cv2.imshow('img2',img2)

                if cv2.waitKey(10) & 0xff == ord("q"):#27
                    break
            else:
                break
        cv2.destroyAllWindows()
        cap.release()

class Kalman(object):
    
    def __init__(self) -> None:
        self.test()

    @staticmethod
    def mousemove(event, x, y, s, p):
        global frame, current_measurement, measurements
        global last_measurement
        global current_prediction, last_prediction

        last_prediction = current_prediction
        last_measurement = current_measurement
        current_measurement = np.array([[np.float32(x)],[np.float32(y)]])
        kalman=kalman()
        kalman.correct(current_measurement)
        current_prediction = kalman.predict()

        lmx, lmy = last_measurement[0], last_measurement[1]
        cmx, cmy = current_measurement[0], current_measurement[1]
        lpx, lpy = last_prediction[0], last_prediction[1]
        cpx, cpy = current_prediction[0], current_prediction[1]

        if event == cv2.EVENT_MOUSEMOVE:
            cv2.line(frame, (int(lmx), int(lmy)), (int(cmx), int(cmy)), (0,100,0))
            cv2.line(frame, (int(lpx), int(lpy)), (int(cpx), int(cpy)), (0,0,100))

    def test(self):
        frame = np.zeros((600, 600, 3), np.uint8)
        last_measurement = current_measurement = np.array((2,1),np.float32)
        last_prediction = current_prediction = np.zeros((2,1), np.float32)


        cv2.namedWindow("kalman_tracker")
        cv2.setMouseCallback("kalman_tracker", Kalman().mousemove)

        kalman = cv2.KalmanFilter(4,2)
        kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)

        kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)

        kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)* 0.03

        while True:
            cv2.imshow("kalman_tracker", frame)
            if (cv2.waitKey(30) & 0xFF) == 27:
                break
        cv2.destroyAllWindows()

"""
def test():
    cap = cv2.VideoCapture(0)
    ret,frame = cap.read()
    r,h,c,w = 150,120,180,150
    track_window = (c,r,w,h)

    roi = frame[r:r+h, c:c+w]
    hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_array=np.array((100., 30.,32.))
    upper_array=np.array((180., 120.,252.))
    mask = cv2.inRange(hsv_roi, lower_array,upper_array)
    
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,1 )
    
    while(1):
        ret ,frame = cap.read()
        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            ret, track_window = cv2.CamShift(dst, track_window,term_crit)
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame,[pts],True, 255,2)
            cv2.imshow('img2',frame)
            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
        else:
            break

    cv2.destroyAllWindows()
    cap.release()
"""