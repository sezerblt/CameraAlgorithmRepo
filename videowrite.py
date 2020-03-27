import cv2
import argparse
import imutils
import time
import numpy
from imutils.video import VideoStream
#from __future__ import print_function
#acces a video sram
#read frames
#construct new Frame
#out to videofile
ap=argparse.ArgumentParser()
ap.add_argument("-o","--output",
                help="cikis video icin PATH belirle")
ap.add_argument("-p","--pycamera",type=int,default=-1,
                help="diger camera hooop ismail kamaragg gidiyyy :)")
ap.add_argument("-f","--fps",type=int,default=24,
                help="FPS orani belirle")
ap.add_argument("-c","--codec",type=str,default="MJPG",
                help="codec formatinda viedo cikisi")
args=vars(ap.parse_args())

print("camera turn on...")
vs=VideoStream(usePiCamera=args["pycamera"] > 0).start()
time.sleep(2.0)
fourcc=cv2.VideoWriter_fourcc(*args["codec"])
writer_mode=None
(h,w)=(None,None)
zeros_mode=None
# döngü: video akışı baslat
while True:
    # çerçeveyi video akısından  tut
    # 360 piksel maksimum genişligi ayarla
    frame = vs.read()
    frame = imutils.resize(frame,width=480)
    #writer mode  kontrolu yap
    if writer_mode is None:
        # video writer basladiginda, boyutları olustur
	# ve sıfır matrisi oluturalım
        h = frame.shape[0]
        w = frame.shape[1]
        writer_mode=cv2.VideoWriter(args["output"],
                                    fourcc,
                                    args["fps"],
                                    (w*2,h*2),
                                    True)
        zeros = numpy.zeros((h,w),dtype='uint8')
        # görüntüyü RGB bileşenlerine ayır
	# her Frame icin RGB degerleri göster
	
    (BLUE,GREEN,RED)=cv2.split(frame)
    RED  = cv2.merge([zeros,zeros,RED])
    GREEN= cv2.merge([zeros,GREEN,zeros])
    BLUE = cv2.merge([BLUE,zeros,zeros])
    #frame tut ve sona kle
    #solust,sagust Red,sagalt Green,solalt BLUE
    result = numpy.zeros((h*2,w*2,3),dtype="uint8")
    result[0:h,0:w]=frame
    result[0:h,w:w*2]=RED
    result[h:h*2,w:w*2]=GREEN
    result[h:h*2,0:w]=BLUE
	#matrisi dosyaya yaz
    writer_mode.write(result)

    key=cv2.waitKey(1) and 0xFF
    #cv2.imshow("Frame",frame)
    cv2.imshow("Output",result)
    if key==ord("q"):
        break

print("Kapatiliyor...")
cv2.destroyAllWindows()
vs.stop()
writer_mode.release()
	



