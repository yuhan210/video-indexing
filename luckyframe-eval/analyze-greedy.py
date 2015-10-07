import numpy as np
import pickle
import os

if __name__ == "__main__":


    VIDEO_LIST = '/mnt/video_list.txt'  
    videos = open(VIDEO_LIST).read().split()

    picked_rates = []
    for video_name in videos:
    
        inter_arrival = []
        file_path = os.path.join('greedy-log', video_name +  '_0.3_gtframe.pickle')
        if not os.path.exists(file_path):
            break
        with open(file_path) as gt_fh:
            selected_obj = pickle.load(gt_fh)
            picked_rates +=  [selected_obj['picked_rate']]
            picked_fid = selected_obj['picked_f']
            n_frames = selected_obj['total_frame'] 
        
            for i in xrange(len(picked_fid) - 1):
                inter_arrival += [picked_fid[i+1] - picked_fid[i]]
            if picked_fid[-1] < (n_frames - 1) - 2:
                inter_arrival += [(n_frames - 1) - picked_fid[-1]]
        print inter_arrival
        
    print np.mean(picked_rates), np.std(picked_rates, ddof = 1)
    print len(picked_rates)
