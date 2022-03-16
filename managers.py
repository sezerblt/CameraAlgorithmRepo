import cv2
import numpy
import time


class WindowManager(object):
    def __init__(self, windowName, keypressCallback = None):
        self.keypressCallback = keypressCallback
        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):
        return self._isWindowCreated    


    def createWindow (self):
        print("\nCREATE WINDOW\n---Creating Window---")
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        print("\nSHOW WINDOW\n---Window---")
        cv2.imshow(self._windowName, frame)
        #cv2.waitKey(0)#for test

    def processEvents (self):
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            # Discard any non-ASCII info encoded by GTK.
            keycode &= 0xFF #pressing 'q' key
            self.keypressCallback(keycode)

    def destroyWindow (self):
        print("\nDESTROY WINDOW\n---Destroyed Window---")
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

#############CAPTURE####################################################################
class CaptureManager(object):
    def __init__(self, capture, previewWindowManager = None,shouldMirrorPreview = False):
        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview
        #private members
        self._capture = capture
        self._channel = 0
        self._enteredFrame = False
        self._frame = None

        self._imageFilename = None
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None
        self._startTime = None
        self._framesElapsed = 0
        self._fpsEstimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve()
        print("our frame: ",self._frame)
        return self._frame

    @property
    def isWritingImage (self):
        print("is writing image?: ",self._imageFilename is not None)
        return self._imageFilename is not None

    @property
    def isWritingVideo(self):
        print("is writing video?: ",self._videoFilename is not None)
        return self._videoFilename is not None

    def enterFrame(self):
        print("""\nENTER\n---Capture the next frame, if any---""")
        # But first, check that any previous frame was exited.
        assert not self._enteredFrame,'previous enterFrame() had no matching exitFrame()'
        if self._capture is not None:
            self._enteredFrame = self._capture#self._enteredFrame = self._capture.grab()
            print("enteredFrame: ",self._enteredFrame)

    def exitFrame (self):
        print("""\nEXIT\n---Draw to the window. Write to files. Release the frame---""")
        # Check whether any grabbed frame is retrievable.
        # The getter may retrieve and cache the frame.
        if self.frame is None:
            self._enteredFrame = False
            return
        # Update the FPS estimate and related variables.
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            self._fpsEstimate = self._framesElapsed / timeElapsed
        self._framesElapsed += 1

        # Draw to the window, if any.
        if self.previewWindowManager is not None:
            print("prev-Wind-Mangr != None")
            if self.shouldMirrorPreview:
                print("mirror preview!")
                mirroredFrame = numpy.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirroredFrame)
            else:
                print("mirror not preview!")
                self.previewWindowManager.show(self._frame)

        # Write to the image file, if any.
        if self.isWritingImage:
            print("wwriting image...")
            cv2.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None
        
        # Write to the video file, if any.
        self._writeVideoFrame()
        # Release the frame.
        self._frame = None
        self._enteredFrame = False
    
    import time
    def writeImage(self, filename):
        print("""\nWRITE IMAGE\n---Write the next exited frame to an image file---""")
        self._imageFilename = filename+str(time.time())

    def startWritingVideo(self, filename,encoding = cv2.VideoWriter_fourcc('I','4','2','0')):
        print("""\START WRITING VIDEO\nStart writing exited frames to a video file.""")
        
        self._videoFilename = filename+str(time.time())
        self._videoEncoding = encoding

    def stopWritingVideo (self):
        print("""\STOP WRITING VIDEO\nStop writing exited frames to a video file.""")
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

    ##private function
    def _writeVideoFrame(self):
        if not self.isWritingVideo:
            return
        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:
                # The capture's FPS is unknown so use an estimate.
                if self._framesElapsed < 20:
                    # Wait until more frames elapse so that the
                    # estimate is more stable.
                    return
                else:
                    fps = self._fpsEstimate
            size = (
                int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            self._videoWriter = cv2.VideoWriter(
                self._videoFilename,
                self._videoEncoding,
                fps, size
            )
        self._videoWriter.write(self._frame)