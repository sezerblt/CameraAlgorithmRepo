from _02_handling.managers import CaptureManager,WindowManager
from _02_handling.filters import FindEdgesFilter,SharpenFilter,BlurFilter,EmbossFilter
import _02_handling.filters as filter
import cv2

class CameraIO(object):
    def __init__(self):
        self._windowManager = WindowManager('Cameraio',self.onKeypress)
        cap=cv2.VideoCapture(0)
        self._captureManager = CaptureManager(cap, self._windowManager, True)
        self._curveFilter = BlurFilter()
        
    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            print(frame[0])
            #filters
            filter.strokeEdges(frame, frame)#**************
            self._curveFilter.apply(frame, frame)
            #
            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress (self, keycode):
        print("""Handle a keypress.
                space -> Take a screenshot.
                tab -> Start/stop recording a screencast.
                escape -> Quit."""
            )
        if keycode == 32: # space
            print("SPACE press")
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9: # tab
            print("TAB press")
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
               self._captureManager.stopWritingVideo()
        elif keycode == 27: # escape
            print("ESC press")
            self._windowManager.destroyWindow()