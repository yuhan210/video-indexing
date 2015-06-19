#!/usr/bin python

import numpy as np
import cv2

# local modules
from utils.video import create_capture
from utils.common import clock, draw_str

import time
import json
import os

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(25, 25), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':

    import sys, getopt

    video_folder = sys.argv[1]
    
    cascade_fn = "./models/haarcascades/haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(cascade_fn)

    for v in os.listdir(video_folder):
        print v        
        
        json_filename = v + '_openfd.json'
        if os.path.exists(os.path.join('/home/t-yuche/frame-analysis/face-info/', json_filename)):
            continue

        blob = {}
        blob['video_name'] = v
        blob['img_blobs'] = []

        frame_folder = os.path.join(video_folder, v)
        for f in os.listdir(frame_folder):
            
            img_blob = {}
            img_blob['img_name'] = f
 
            img_path = os.path.join(frame_folder, f)            
            img = cv2.imread(img_path)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            tic = time.time()
            rects = detect(gray, cascade)
            toc = time.time()
            faces = []
            for x1, y1, x2, y2 in rects:
                faces += [([int(x1), int(y1), int(x2-x1), int(y2-y1)])]
            
            #print faces    
            img_blob['faces'] = faces
            img_blob['fd_time'] = (toc-tic)
            blob['img_blobs'] += [img_blob]
        
        json_filename = v + '_openfd.json'
        json.dump(blob, open(os.path.join('/home/t-yuche/frame-analysis/face-info/', json_filename), 'w'))
