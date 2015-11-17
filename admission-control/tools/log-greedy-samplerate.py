import numpy as np
import math
import pickle
import sys
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":


    THRESHS = [1]
    VIDEO_LIST = '/mnt/video_list.txt'  
    videos = open(VIDEO_LIST).read().split()

    for THRESH in THRESHS:
        picked_rates = []
        inter_arrivals = []
        for video_name in videos:

            file_path = os.path.join('../window-greedy-log', video_name +  '_' + str(THRESH) + '_gtframe.pickle')

            with open(file_path) as gt_fh:
                selected_obj = pickle.load(gt_fh)
                picked_rate = selected_obj['picked_rate']
                picked_rates += [picked_rate]
                picked_fid = selected_obj['picked_f']
                n_frames = selected_obj['total_frame'] 
            
                assert(picked_rate == len(picked_fid)/(n_frames * 1.0)) 

        print THRESH, np.mean(picked_rates), np.std(picked_rates, ddof = 1)
