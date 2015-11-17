from vision import *
import numpy as np
import pickle
import cv2
import time

if __name__ == "__main__":

    VIDEO_LIST = '/mnt/video_list.txt'
    GREEDYFOLDER = '/home/t-yuche/admission-control/window-greedy-log'
    BTFCVINFOFOLDER = '/mnt/tags/cv-btframe-08'
    MV_INPUT_FOLDER = '/home/t-yuche/admission-control/train/mv_log'
    
    GREEDYTHRESH = 0.8
    videos = open(VIDEO_LIST).read().split()

    outfile_path = open('train.data', 'w')
    for video_name in videos:
        print video_name

        tic = time.time()
        frames = sorted(os.listdir(os.path.join('/mnt/frames', video_name)), key = lambda x: int(x.split('.')[0]))
        toc = time.time()
        
        print 'sort frame time', toc - tic
        # read greedy trace
        greedypath = os.path.join(GREEDYFOLDER, video_name + '_' + str(GREEDYTHRESH)  + '_gtframe.pickle')
        tic = time.time()
        gt_data = pickle.load(open(greedypath))
        toc = time.time()
        print 'load greedy time', toc - tic
        gt_picked_fid = gt_data['picked_f']
        total_frame_n = gt_data['total_frame']

        # feature before decoding
        tic = time.time()
        encdata = getMetadata(video_name) 
        mv_file = os.path.join(MV_INPUT_FOLDER, video_name + '.pickle')
        with open(mv_file) as fh:
            mv_features = pickle.load(fh)
        toc = time.time()
        print 'load compressed time', toc - tic
        w = encdata['metadata']['w'] 
        h = encdata['metadata']['h']

        # features after decoding
        cvdata = getCVInfoFromLog(video_name)
        btfdata = getBTFrameInfoFromLog(video_name, BTFCVINFOFOLDER) 

        for idx, frame_name in enumerate(frames):
            fid = int(frame_name.split('.')[0])
            assert(idx == fid)
            if fid == 0:
                continue
            #print mvdata 
            enc = encdata[frame_name]  
            cv = cvdata[frame_name]
            btf_cv = btfdata[frame_name]

            label = 0
            if fid in gt_picked_fid:
                label = 1
            # write features
            feature_str = video_name + ','  + str(fid) + ',' + enc['type'] + ',' +  str(w) + ',' + str(h) + ',' + str(enc['size']) + ',' + str(mv_features[fid][0]) + ',' + str(mv_features[fid][1]) + ',' + str(mv_features[fid][2]) + ',' + str(mv_features[fid][3]) +  ',' + str(cv['sobel'][0]) + ',' + str(cv['illu'][0]) + ',' +str(btf_cv['framediff'][0]) + ',' + str(btf_cv['phash'][0]) + ','+ str(btf_cv['colorhist'][0]) +','+ str(btf_cv['siftmatch'][0]) +','+ str(btf_cv['surftime'][0]) +','+ str(label)
            outfile_path.write(feature_str + '\n')
            outfile_path.flush()
    outfile_path.close()
