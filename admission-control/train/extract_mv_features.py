from vision import *
import pickle
import sys
import os
import numpy as np

if __name__ == "__main__":

    pvid = int(sys.argv[1]) -1
    VIDEO_LIST = '/mnt/video_list.txt'
    videos = open(VIDEO_LIST).read().split()
    MV_OUTPUT_FOLDER = '/home/t-yuche/admission-control/train/mv_log'

    for vid, video_name in enumerate(videos):
        if pvid != vid:
            continue
        mvdata, encdata = getNormCompressedInfoFromLog(video_name)
        mv_file = os.path.join(MV_OUTPUT_FOLDER, video_name + '.pickle')
        mv_feature = {}
        frames = sorted(os.listdir(os.path.join('/mnt/frames', video_name)), key = lambda x: int(x.split('.')[0]))
        print video_name        
        for idx, frame_name in enumerate(frames): 
            fid = int(frame_name.split('.')[0])
            frame_name_mv = str(fid + 1) + '.jpg'
            if frame_name_mv in mvdata:
                mvs = mvdata[frame_name_mv]
            else:
                mvs = [-1]

            m = np.mean(mvs) 
            ma = max(mvs)
            mi = min(mvs)

            mv_feature[fid] = (len(mvs), m, ma, mi)

        with open(mv_file, 'wb') as fh:
            pickle.dump(mv_feature, fh)
