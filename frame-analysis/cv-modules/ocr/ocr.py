import os
import sys
import json
import time
import Image
import pytesseract


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
       
        tic = time.time() 
        ocr_str = pytesseract.image_to_string(Image.open(os.path.join(frame_folder, f)))
        toc = time.time()
        
        img_blob['ocr'] = ocr_str
        img_blob['ocr_time'] = toc - tic
        blob['img_blobs'] += [img_blob]
    
    json_filename = v + '_ocr.json'
    json.dump(blob, open(os.path.join('/home/t-yuche/frame-analysis/ocr-info/', json_filename), 'w'))
