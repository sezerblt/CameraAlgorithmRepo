import datetime
import argparse
import imutils
import time
import cv2
from threading import Thread

class WebcamVideoStream:
    def __init__(self,src=0):
        self.stream = cv2.VideoCapture(src)
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

class VideoStream:
	def __init__(self, src=0, usePiCamera=False, resolution=(320, 240),
		framerate=32):
		# pycam modulu ?*
		if usePiCamera:
			# gerekli kurulum
			from pivideostream import PiVideoStream
			self.stream = PiVideoStream(resolution=resolution,
				framerate=framerate)
		# else
		else:
			self.stream = WebcamVideoStream(src=src)
	def start(self):
		# thread video stream start
		return self.stream.start()
	def update(self):
		#  frame video stream
		self.stream.update()
	def read(self):
		#curent frame dondur
		return self.stream.read()
	def stop(self):
		#
		self.stream.stop()

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	timestamp = datetime.datetime.now()
	ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
	cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		0.35, (0, 0, 255), 1)
	# show methhoud frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# q basılınca
	if key == ord("q"):
		break
#
cv2.destroyAllWindows()
vs.stop()
