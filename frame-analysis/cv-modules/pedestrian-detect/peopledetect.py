#!/usr/bin/env python

import numpy as np
import cv2
import os
import time
import json

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


if __name__ == '__main__':
    import sys


    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

    video_folder = sys.argv[1]
    for v in os.listdir(video_folder):
        print v

        blob = {}
        blob['video_name'] = v
        blob['img_blobs'] = []

        frame_folder = os.path.join(video_folder, v)
        for f in os.listdir(frame_folder):

            img_blob = {}
            img_blob['img_name'] = f
            img_path = os.path.join(frame_folder, f)
            
            img = cv2.imread(img_path)
            
            tic = time.time()        
            found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
            toc = time.time()

            '''
            found_filtered = []
            for ri, r in enumerate(found):
                for qi, q in enumerate(found):
                    if ri != qi and inside(r, q):
                        break
                    else:
                        found_filtered.append(r)
            ''' 
            people = []
            for x, y, w, h in found:
                 people += [ [int(x), int(y), int(w), int(h)] ]

            img_blob['peds'] = people
            img_blob['pd_time'] = toc - tic
            blob['img_blobs'] += [img_blob]

        json_filename = v + '_openpd.json'
        json.dump(blob, open(os.path.join('/home/t-yuche/frame-analysis/ped-info/', json_filename), 'w'))
