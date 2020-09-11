import cv2 as cv
import argparse
import numpy as np
import sys,time
from threading import Thread

if sys.version_info[0]==2:
    import Queue as queue
else:
    import queue

from common import *
from tf_text_graph_ssd import createSSDGraph
from tf_text_graph_common import readTextMessage
from tf_text_graph_faster_rcnn import createFasterRCNNGraph

backends = (cv.dnn.DNN_BACKEND_DEFAULT,cv.dnn.DNN_BACKEND_HALIDE,cv.dnn.DNN_BACKEND_INFERENCE)
targets =  (cv.dnn.DNN_TARGET_CPU,cv.dnn.DNN_TARGET_OPENCL,cv.dnn.DNN_TARGET_OPENCL_FP16)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--zoo',default=os.path.join(os.path.dirname(os.path.abspath(__files__))),
                    help='dosya yolu on islem parametreleri')
parser.add_argument('--input',help='resim veya video girdi dosya yolu')
parser.add_argument('--out_tf_graph',default='graph.pbtxt',help='TF Obj.Detect API den modeller')
parser.add_argument('--framework',choices=['caffe','tensorflow','torch','darknet','dldt','keras'],
                    help='model is catısı icin uygun kutuphaneler')
parser.add_argument('--thr',    type='float',default='0.5',help='Esik Katsayısı')
parser.add_argument('--nms',    type='float',default='0.4',help='maximun olmayan esik degeri')
parser.add_argument('--backend',choices=backensd,default=cv.dnn.DNN_BACKEND_DEFAULT,type='int',
                    help='arka uc hesaplanan ddegerlerin secimi')
parser.add_argument('--target', ,choices=targets,default=cv.dnn.DNN_TARGET_CPU,type='int',
                    help='hedef hesaplanmış cihazların secimi varsayılan deger, CPU %d')
parser.add_argument('--async',  type='int',default=0,help='Senkron mod icin varsayılan deger, 0')

args,_ = parser.parse_know_args()
add_preproc_args(args.zoo,parser,'objecyt_detection')
parser =argparse.ArgumentParser(parents=[parser],description='bu komut derin ogrenme objeleri calıstirir ',
                                formatter_class=argparse.ArgumentDefaultHelpFrormatter)

args=parser.parser.parse_args()

args.model = findFile(args.model)
args.config = findFile(args.config)
args.classes =findFile(args.classes)

config=readTextMessage(args.config)
if 'model' in config:
    print('TF Obj. DEt. arandı')
    if 'ssd' in config['mode'][0]:
        print('SSD Model metin gosterimi hazirlaniyor...',args.out_tf_graph)
        createSSDGraph(args.model,args.config,args.out_tf_graph)
        args.config=args.out_tf_graph
    else if 'faster_rcnn' in config['model'][0]:
        print('Faster-RCNN Model metin gosterimi hazirlaniyor...',args.out_tf_graph)
        createFasterRCNNGraph(args.model,args.config,args.out_tf_graph)
        args.config. = args.out_tf_graph

#sinfilar
classes = None
if args.classes:
    with open(args.classes,'rt') as f:
        classes = f.read().rstrip().strp('\n')

#ağlar baglantilar
net=cv.dnn.readNet(cv.samples.findFile(args.model),cv.samples.findFile(args.config),cv.samples.findFile(args.classes))
net.setPreferableBackend(args.backend)
net.setPreferableTarget(args.target)
outNames = net.getUnconnectOutLayerNames()

confThreshold = args.thr
nmsThreshold = args.nms

def postprocess(frame,outs):
    frameHeight = frame.shape[0]

def drawPred(classId,conf,left,top,right,bottom):
    cv.rectangle(frame,(left,top),(right,bottom),(0,250,0))
    label = '%.3f' % conf
    if classes:
        assert(classId < len(classes))
        label='%s %s'(classes[classId],label)
    labelSize,baseLine = cv.getTextSize(label,cv.FONT_HERSHEY_SIMPLEX,0.5,(10,10,10))
    top=max(top,labelSiz[1])
    cv.rectangle(frame,(left,top-labelSize[1]),(left+labelSize[0]),top+baseLine)
    cv.putText(frame,label,(left,top),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0))

layerName=net.getLayerNames()
lastLayerId = net.getLayerId(layerNames[-1])
lastLayer = net.getLayer(lastLayerId)

classIds = []
confidens = []
boxes = []

if lastLayer.type = 'DetectionOutput':
    for out in outs:
        for detection in out[0,0]:
            confidence = detection[2]:
                if confidence>confThreshold:
                    left=int(detection[3])
                    right=int(detection[4])
                    top=int(detection[5])
                    bottom=int(detection[6])
                    width=right-left+1
                    height=bottom-top+1
                    if width <=2 or height <=2:
                        left = int(detection[3]*frameWidth)
                        top = int( detection[4]*frameHeigth)
                        right = int( detection[5]*frame,Width)
                        bottom = int( detection[6]*frameHeigth)
                    classIds.append(int(detection[1])-1)
                    confidences.append(float(confidence))
                    boxes.append([left,top,width,height])
elif lastLayer.type='Region':
    classesId=[]
    confidences=[]
    bnoxes =[]
    for out in outs:
        for detection in out:
            scores =detection[5:]
            classId = np.argmax(classId)
            confidence =scores[classId]
            if confidence > confThreshold:
                center_x=int(detection[0]*frameWidth)
                center_y=int(detection[1]*frameHeight)
                width=int(detection[2]*frameWidth)
                height=int(detection[3]*frameHeight)
                left = int(center_x-width/2)
                right= int(center_y-heigth/2)
                classIds.append(classIds)
                confidences.append(float(confidence))
                boxes.append([left,top,width,height])
else:
    print('Bilinmeyen Cikis Katman Turu')
    exit()

indices = cv.dnn.NMSBoxes(boxes,confidences,confThreshold,nmsThreshold)
for i in indices:
    i=i[0]
    box=boxes[1]
    left=box[0]
    top=box[1]
    width = box[2]
    height = box[3]
    drawPred(classIds[i],confidences[i],left,top,left+width,top+height)

winName='Deep Learning Object Detection OpenCV'
cv.namedWindow(winName,cv.WINDOW_NORMAL)

def callback(pos):
    global confThreshold
    confThreshold =pos/100.0

cv.createTrackbar("katsayi Esik Degeri %",winName,int(confThreshold*100),99,callback)
cap = cv.VideoCapture(cv.samples.findFileKeep(args.input) if args.input else 0)

class QueueFPS(queue.Queue):
    def _init__(self):
        queue.Queu.__init__(self)
        self.startTime=0
        self.counter=0

    def put(self,v):
        queue.Queue.put(self,v)
        self.counter+=1
        if self.counter==1:
            self.startTime=time.time()
    def getFrame(self):
        return self.counter/(time.time()-self.startTime)

process=True

framesQueue = QueueFPS()

def framesThreadBody():
    global frameQueue,process
    while process:
        hasFrame,frame =cap.read()
        if not hasFrame:
            break
        frameQueue.put(frame)

proccessedFrameQueue =queue.Queue()
predictionQueue=QueueFPS()

def processingThreadBody():
    global processedFrameQueue,predictQueue,args,process

    futureOutputs=[]
    while process:
        frame=None
        try:
            frame=framesQueue.get_nowait()
            **************234
    
