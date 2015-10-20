from vision import *
import pickle
import time
import os
import cv2

if __name__ == "__main__":

    VIDEO_LIST = '/mnt/video_list.txt'
    CVFOLDER = '/mnt/tags/cv-info'
    videos = open(VIDEO_LIST).read().split()

    for video in videos:
        print video
        data = {}
        frames = sorted(os.listdir(os.path.join('/mnt/frames', video)), key = lambda x: int(x.split('.')[0]))
        for frame_name in frames:
            img = cv2.imread(os.path.join('/mnt/frames', video, frame_name))    
            h, w = img.shape[:2]
            if  h * w > 320 * 240:
                img = cv2.resize(img, (320, 240)) 
            
            tic = time.time()
            sobel = getSobel(img)
            toc = time.time()
            sobeltime = toc - tic
            tic = time.time()
            illu = getIlluminance(img)
            toc = time.time()
            illutime = toc - tic
            data[frame_name] = {'sobel': [sobel, sobeltime], 'illu': [illu, illutime]}
    
        with open(os.path.join(CVFOLDER, video + '.pickle'), 'wb') as fh:
            pickle.dump(data, fh) 

