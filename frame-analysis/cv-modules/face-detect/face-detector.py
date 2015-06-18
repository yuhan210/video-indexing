#!/usr/bin/python
import os
import sys

import dlib
import json
import time
from skimage import io


video_folder = sys.argv[1]
detector = dlib.get_frontal_face_detector()

for v in os.listdir(video_folder):
    print v
    
    blob = {}
    blob['video_name'] = v
    blob['img_blobs'] = []

    frame_folder = os.path.join(video_folder, v)

    for f in os.listdir(frame_folder):
        img_blob = {}
        img_blob['img_name'] = f
        img = io.imread(os.path.join(frame_folder, f))

        # The 1 in the second argument indicates that we should upsample the image
        # 1 time.  This will make everything bigger and allow us to detect more
        # faces.
        tic = time.time()
        dets, scores, idx = detector.run(img, 1)
        toc = time.time()
        faces = []
        for i, d in enumerate(dets):
            faces += [([d.left(), d.top(), d.right() - d.left(), d.bottom() - d.top()], scores[i], idx[i])]        

        img_blob['faces'] = faces
        img_blob['fd_time'] = (toc-tic) 
        blob['img_blobs'] += [img_blob]

    json_filename = v + '_dlibfd.json'
    json.dump(blob, open(os.path.join('/home/t-yuche/frame-analysis/face-info/', json_filename), 'w'))

