from vision import *
import pickle
import time
import os
from PIL import Image
if __name__ == "__main__":

    VIDEO_LIST = '/mnt/video_list.txt'
    GREEDYFOLDER = '/home/t-yuche/admission-control/window-greedy-log'
    GREEDYTHRESH = 0.8
    CVFOLDER = '/mnt/tags/cv-btframe-08'
    videos = open(VIDEO_LIST).read().split()

    for video_name in videos:
        print video_name
        outfilepath = os.path.join(CVFOLDER, video_name + '.pickle')
        if os.path.exists(outfilepath):
            continue

        # read greedy trace
        greedypath = os.path.join(GREEDYFOLDER, video_name + '_' + str(GREEDYTHRESH)  + '_gtframe.pickle')
        gt_data = pickle.load(open(greedypath))
        gt_picked_fid = gt_data['picked_f']
        total_frame_n = gt_data['total_frame']

        data = {}
        frames = sorted(os.listdir(os.path.join('/mnt/frames', video_name)), key = lambda x: int(x.split('.')[0]))
        prev_fid = 0
        for framename in frames:
            fid = int(framename.split('.')[0])
            if fid == 0:
                continue
            
            prev_framename = str(prev_fid) + '.jpg'

            cur_img = cv2.imread(os.path.join('/mnt/frames', video_name, framename))     
            prev_img = cv2.imread(os.path.join('/mnt/frames', video_name, prev_framename))     

            h, w = cur_img.shape[:2]
            if  h * w > 320 * 240:
                cur_img = cv2.resize(cur_img, (320, 240)) 
                prev_img = cv2.resize(prev_img, (320, 240)) 

            # frame diff
            tic = time.time()
            framediff = getFrameDiff(prev_img, cur_img)
            toc = time.time()
            framediff_time = toc - tic
            framediff_prec = framediff/ (h * w * 1.0)

            # phash
            pilcur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
            pilprev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)
            pil_cur = Image.fromarray(pilcur_img)
            pil_prev = Image.fromarray(pilprev_img)
            tic = time.time()
            phash_v = phash(pil_prev, pil_cur) 
            toc = time.time()
            phash_time = toc - tic

            # color histogram
            tic = time.time()
            hist_score = colorHistSim(prev_img, cur_img)
            toc = time.time()
            hist_time = toc - tic

            # SIFT + point matching
            sprev_img = cv2.resize(prev_img, (160,120))
            scur_img = cv2.resize(cur_img, (160,120))
            tic = time.time()
            sift_score = getSIFTMatchingSim(sprev_img, scur_img)            
            toc = time.time()
            sift_time = toc - tic

            tic = time.time()
            surf_score = getSURFMatchingSim(sprev_img, scur_img)            
            toc = time.time()
            surf_time = toc - tic

            data[framename] = {'framediff': [framediff_prec, framediff_time], 'phash': [phash_v, phash_time], 'colorhist': [hist_score, hist_time], 'siftmatch': [sift_score, sift_time],'surftime': [surf_score, surf_time]}
            if fid in gt_picked_fid:
                prev_fid = fid

        with open(os.path.join(CVFOLDER, video_name + '.pickle'), 'wb') as fh:
            pickle.dump(data, fh) 

