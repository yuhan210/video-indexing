import os
import cv2
import json
import math
import numpy as np
import pickle
from PIL import Image
import imagehash

def parseMVs(video_name, folder = '/mnt/tags/video-encoding-info'):

    mvdata = {}
    with open(os.path.join(folder, video_name + '_mvs.txt')) as fh:
        for idx, line in enumerate(fh.readlines()):
            if idx == 0:
                continue
            segs = line.split(',')
            segs = [x.strip() for x in segs]
            framename = segs[0] + '.jpg'
            source = int(segs[1])
            blockw = int(segs[2])
            blockh = int(segs[3])
            srcx = int(segs[4])
            srcy = int(segs[5])
            dstx = int(segs[6])
            dsty = int(segs[7])

            dist = math.sqrt((srcx - dstx)**2 + (srcy - dsty)**2)
            if framename not in mvdata:
                mvdata[framename] = []
            mvdata[framename] += [dist]

    return mvdata

def parseVideoProbeLog(video_name, folder = '/mnt/tags/video-encoding-info'):

    data = {}
    with open(os.path.join(folder, video_name + '_encoding.json')) as fh:
        raw_data = json.load(fh)
        for x in raw_data['packets_and_frames']:
            if x['type'] == 'frame':
                framenum = x['coded_picture_number']
                
                frame_name = str(framenum) + '.jpg'
                print frame_name
                w = int(x['width'])
                h = int(x['height'])
                frame_type = x['pict_type'] # I, P, B
                size = int(x['pkt_size'])
                
                if 'metadata' not in data:
                    data['metadata'] = {'w': w, 'h':h} 
                data[frame_name] = {'type': frame_type, 'size':size}

    return data

if __name__ == "__main__":


    VIDEO_LIST = '/mnt/video_list.txt'
    PROCESSED_FOLDER = '/mnt/tags/cv-processed-info'
    for video_name in open(VIDEO_LIST).read().split():
        
        mv_data = parseMVs(video_name)
        encoding_data = parseVideoProbeLog(video_name)

        with open(os.path.join(PROCESSED_FOLDER, video_name '_mv.pickle'), 'wb') as fh:
            pickle.dump(mv_data, fh)

        with open(os.path.join(PROCESSED_FOLDER, video_name '_enc.pickle'), 'wb') as fh:
            pickle.dump(encoding_data, fh)

def getCVInfoFromLog(video_name, folder = '/mnt/tags/cv-info'):
    
    with open(os.path.join(folder, video_name)) as fh:
        cvdata = pickle.load(fh)

    return cvdata

def getSobel(img, k_size = 3):

    ddepth = cv2.CV_16S
    scale = 1
    delta = 0

    cv2.GaussianBlur(img, (3,3), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Gradient-x
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize = k_size, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)

   
    #Gradient-y
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize = k_size, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)

    # converting back to uint8
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
   
    dst = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
    #dst = cv2.add(abs_grad_x,abs_grad_y)

    return dst.mean()

def getIlluminance(img):

    img_out = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y,u,v = cv2.split(img_out) 
    ave_lum = np.mean(y)
    
    return ave_lum

def phash(a, b):
    a_hash = imagehash.phash(a)
    b_hash = imagehash.phash(b)

    return a_hash - b_hash

def dhash(a, b):
    a_hash = imagehash.dhash(a)
    b_hash = imagehash.dhash(b)

    return a_hash - b_hash

def ahash(a, b):
    a_hash = imagehash.average_hash(a)
    b_hash = imagehash.average_hash(b)
    
    return a_hash - b_hash
