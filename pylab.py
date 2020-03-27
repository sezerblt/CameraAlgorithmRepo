import datetime
from threading import Thread
import cv2
import argparse
import imutils


class FPS:
    def __init__(self):
        self._start= None
        self._end  = None
        self._numberFrames = 0

    def start(self):
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        self._stop = datetime.datetime.now()

    def update(self):
        self._numberFrames +=1

    def elapsed(self):
        return (self._end - self._start).total_seconds()

    def fps(self):
        return self._numberFrames/self.elapsed()

class WebcamVideoStream:
    def __init__(self,source=0):
        self.stream = cv2.VideoCapture(source)
        (self.grabbed,self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update,args=()).start()

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed,self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped=True
            
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-n","--numframe",type=int,
                        default=100,
                        help="fps test icin frame sayisi")
arg_parser.add_argument("-d","--display",type=int,
                        default=-1,
                        help="fps test icin goruntuleme")
args=vars(arg_parser.parse_args())
print("(DURUM)web kamerasindan framme aliniyor")
stream = cv2.VideoCapture(0)
fps    = FPS().start()

while fps._numberFrames<args["numframe"]:
    grab,frame = stream.read()
    frame = imutils.resize(frame,width=360)

    if args["display"]>0:
        cv2.imshow("Frame",frame)
        key = cv2.waitKey(1) and 0xFF
    fps.update()
fps.stop()
print("geciken sure:{:.2f}".format(fps.elapsed()))
print("FPS :{:.2f}".format(fps.fps()))

strem.release()
cv2.destroyAllWindows()
#--------------------
print("web cam thead:")
vs  = WebcamVideoStream(source=0).start()
fps = FPS().start()

while fps._numberFrames<args["numberframes"]:
    frame2 = vs.read()
    frame2 = imutils.resize(frame2,width=380)

    if args["display"]>0:
        cv2.imshow("Frame 2",frame)
        key = cv2.waitKey(1) and 0xFF
    fps.update()
    
fps.stop()
print("geciken sure:{:.2f}".format(fps.elapsed()))
print("FPS :{:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
v.stop()
