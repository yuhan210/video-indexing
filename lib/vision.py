import os
import cv2
import json
import math
import numpy as np
import pickle
from PIL import Image
import imagehash

def get_video_fps(video_name):

    video_path = os.path.join('/mnt/videos', video_name)
    if video_path.find('.mp4') < 0:
        video_path += '.mp4'

    cap = cv2.VideoCapture(video_path)    
    fps  = cap.get(cv2.cv.CV_CAP_PROP_FPS) 
    w  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
 
    return fps, w, h

def colorHistSim(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)

    ahist = cv2.calcHist([a], [0, 1], None, [180, 256], [0, 180, 0, 256])
    bhist = cv2.calcHist([b], [0, 1], None, [180, 256], [0, 180, 0, 256])
    
    cv2.normalize(ahist,ahist,0,255,cv2.NORM_MINMAX)
    cv2.normalize(bhist,bhist,0,255,cv2.NORM_MINMAX)

    sim = cv2.compareHist(ahist, bhist, cv2.cv.CV_COMP_CORREL)

    return sim

def filter_matches(matches):
    good = []
    for m in matches:
        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
            good.append([m[0]])
    return good


def getSIFTMatchingSim(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)    
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)    

    sift = cv2.SIFT()
    kp1, des1 = sift.detectAndCompute(a, None)
    kp2, des2 = sift.detectAndCompute(b, None)
    if len(kp1) == 0 or len(kp2) == 0:
        return 0

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k = 2)
    good_matches = filter_matches(matches)

    return len(good_matches)/(len(kp1) * 1.0)


def getSURFMatchingSim(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)    
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)    

    surf = cv2.SURF()
    kp1, des1 = surf.detectAndCompute(a, None)
    kp2, des2 = surf.detectAndCompute(b, None)
    if len(kp1) == 0 or len(kp2) == 0:
        return 0
     
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k = 2)
    good_matches = filter_matches(matches)
    
    return len(good_matches)/(len(kp1) * 1.0)

def parseMVs_normalized(video_name, folder = '/mnt/tags/video-encoding-info'):

    fps, w, h = get_video_fps(video_name)
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

            x_mov = (dstx - srcx) / (w * 1.0)
            y_mov = (dsty - srcy) / (h * 1.0)
            dist = math.sqrt(x_mov**2 + y_mov**2)
            if framename not in mvdata:
                mvdata[framename] = []
            mvdata[framename] += [dist]

    return mvdata

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

def frameType1ofKCoding(frame_type_str):
    
    if frame_type_str == 'I':
        return (0,0)
    elif frame_type_str == 'P':
        return (0,1)
    elif frame_type_str == 'B':
        return (1,0)
    else:
        print 'Unknown frame type:', frame_type_str
        return None  

def parseVideoProbeLog(video_name, folder = '/mnt/tags/video-encoding-info'):

    data = {}
    with open(os.path.join(folder, video_name + '_encoding.json')) as fh:
        raw_data = json.load(fh)
        for x in raw_data['packets_and_frames']:
            if x['type'] == 'frame':
                framenum = x['coded_picture_number']
                
                frame_name = str(framenum) + '.jpg'
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

        print video_name 
        mvpath = os.path.join(PROCESSED_FOLDER, video_name + '_mv_norm.pickle')
        if os.path.exists(mvpath):
            continue

        mv_data = parseMVs_normalized(video_name)

        with open(mvpath, 'wb') as fh:
            pickle.dump(mv_data, fh)


def getBTFrameInfoFromLog(video_name, folder = '/mnt/tags/cv-btframe-08'):
    #data[frame_name] = {'framediff': [framediff_prec, framediff_time], 'phash': [phash_value, phash_time], 'colorhist': [hist_score, hist_time], 'siftmatch': [sift_score, sift_time], 'surftime': [surf_score, surf_time]}

    with open(os.path.join(folder, video_name + '.pickle')) as fh:
        btfdata = pickle.load(fh)

    return btfdata

def getCVInfoFromLog(video_name, folder = '/mnt/tags/cv-info'):
    #data[frame_name] = {'sobel': [sobel, sobeltime], 'illu': [illu, illutime]}
    #exectime in res. max( 320 x 240, original size)

    with open(os.path.join(folder, video_name + '.pickle')) as fh:
        cvdata = pickle.load(fh)

    return cvdata

def getCompressedInfoFromLog(video_name, folder = '/mnt/tags/cv-processed-info'):
    with open(os.path.join(folder, video_name + '_mv.pickle')) as fh:
        mvdata = pickle.load(fh)
    
    with open(os.path.join(folder, video_name + '_enc.pickle')) as fh:
        encdata = pickle.load(fh)

    # if a frame is not in mvdata. there's no motion vector
    return mvdata, encdata

def getMetadata(video_name, folder = '/mnt/tags/cv-processed-info'):
    
    with open(os.path.join(folder, video_name + '_enc.pickle')) as fh:
        encdata = pickle.load(fh)
    return encdata


def getNormCompressedInfoFromLog(video_name, folder = '/mnt/tags/cv-processed-info'):
    # This is slow
    # suggestion - load from:
    # MV_OUTPUT_FOLDER = '/home/t-yuche/admission-control/train/mv_log'
    #mv_file = os.path.join(MV_OUTPUT_FOLDER, video_name + '.pickle')
    #with open(mv_file, 'r') as fh:
    #mv_features = pickle.load(fh)

    # and use getMetadata
    with open(os.path.join(folder, video_name + '_mv_norm.pickle')) as fh:
        mvdata = pickle.load(fh)
    
    with open(os.path.join(folder, video_name + '_enc.pickle')) as fh:
        encdata = pickle.load(fh)

    return mvdata, encdata

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

def getFrameDiff(prev_frame, cur_frame):

    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    
    frameDelta = cv2.absdiff(prev_frame, cur_frame)
    thresh = cv2.threshold(frameDelta, 35, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)
    movement = cv2.countNonZero(thresh)


    return movement

    

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
