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
		# check to see if the picamera module should be used
		if usePiCamera:
			# only import the picamera packages unless we are
			# explicity told to do so -- this helps remove the
			# requirement of `picamera[array]` from desktops or
			# laptops that still want to use the `imutils` package
			from pivideostream import PiVideoStream
			# initialize the picamera stream and allow the camera
			# sensor to warmup
			self.stream = PiVideoStream(resolution=resolution,
				framerate=framerate)
		# otherwise, we are using OpenCV so initialize the webcam
		# stream
		else:
			self.stream = WebcamVideoStream(src=src)
	def start(self):
		# start the threaded video stream
		return self.stream.start()
	def update(self):
		# grab the next frame from the stream
		self.stream.update()
	def read(self):
		# return the current frame
		return self.stream.read()
	def stop(self):
		# stop the thread and release any resources
		self.stream.stop()

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())
# initialize the video stream and allow the cammera sensor to warmup
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	# draw the timestamp on the frame
	timestamp = datetime.datetime.now()
	ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
	cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		0.35, (0, 0, 255), 1)
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
