import numpy as np
import math
import pickle
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":


    VIDEO_LIST = '/mnt/video_list.txt'  
    videos = open(VIDEO_LIST).read().split()

    picked_rates = []
    inter_arrivals = [] 
    for video_name in videos:
    
        inter_arrival = []
        file_path = os.path.join('window-greedy-log', video_name +  '_0.8_gtframe.pickle')
        if not os.path.exists(file_path):
            break
        with open(file_path) as gt_fh:
            selected_obj = pickle.load(gt_fh)
            picked_rates +=  [selected_obj['picked_rate']]
            picked_fid = selected_obj['picked_f']
            n_frames = selected_obj['total_frame'] 
       
            print video_name, selected_obj['picked_rate'] 
            for i in xrange(len(picked_fid) - 1):
                inter_arrival += [picked_fid[i+1] - picked_fid[i]]
            if picked_fid[-1] < (n_frames - 1) - 2:
                inter_arrival += [(n_frames - 1) - picked_fid[-1]]
        #print inter_arrival
        inter_arrivals += [[np.mean(inter_arrival), np.std(inter_arrival, ddof = 1)/math.sqrt(len(inter_arrival)), 1/(float(selected_obj['picked_rate']) * 1.0)] ]       
 
    print np.mean(picked_rates), np.std(picked_rates, ddof = 1)
    #print inter_arrivals
    plt.errorbar(range(len(inter_arrivals)), [x[0] for x in inter_arrivals], yerr = [x[1] for x in inter_arrivals], fmt='o', color='b', label='greedy') 
    plt.errorbar(range(len(inter_arrivals)), [x[2] for x in inter_arrivals], yerr = [0 for x in inter_arrivals], fmt='x', color='r', label='uniform') 
    plt.ylabel('Inter frame time (frames)')
    plt.xlabel('Video ID')
    plt.legend() 
    plt.show()
