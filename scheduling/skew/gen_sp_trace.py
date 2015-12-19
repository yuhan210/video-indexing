from utils import *
import pickle
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
try:
    plt.style.use('ggplot')
except:
    pass

KS = [3,5,7,9]
def load_all_words():
    with open('dictionary.pickle') as fh:
        return pickle.load(fh)


def match_measure(al, bl):
  
    n_inter = 0
    n_union = 0 
    for w in al:
        if w in bl:
            n_inter += 1
    n_union = len(al) + len(bl) - n_inter
   
    if n_union == 0:
        return -1
 
    return n_inter/(n_union * 1.0) 

if __name__ == "__main__":

    #OPTIMAL_INPUT_FOLDER = '/home/t-yuche/admission-control/eval/optimal-log/optimal-tf-less-stopword'
    OPTIMAL_INPUT_FOLDER = '/home/t-yuche/admission-control/eval/optimal-log/optimal-tf'
    VIDEO_LIST = '/mnt/video_list.txt'
    SERVER_STORAGE_FRAMES = 5 * 30 # 5 sec * 30 fps
    SLIDE_SIZE_FRAMES = 1 * 30 # 1 sec * 30 fps
    UNIMPORTANTWORD_THRESH = 0.5
    TRANSITION_THRESH = 0.2
    ###
    C_N = 10
    C_P = 0.9
    TRACE_LOG = './sp_trace/N_' + str(C_N) + '_P_' + str(C_P)
    
    if not os.path.exists(TRACE_LOG):
        os.makedirs(TRACE_LOG)

    videos = open(VIDEO_LIST).read().split()
    count = 0
    for video_name in videos:

        print video_name
        n_frames = get_video_frame_num(video_name)
        
        with open(os.path.join(OPTIMAL_INPUT_FOLDER, video_name + '_' + str(UNIMPORTANTWORD_THRESH) + '.pickle')) as fh:
            optimal_data = pickle.load(fh)
     

        start_fid = 0
        prev_tf = {}
        sp_log = []
        sp_trace = {}
        while True:
            
            if start_fid > n_frames - 1:
                break
        
            end_fid = min(start_fid + SERVER_STORAGE_FRAMES, n_frames) 
            key = str(start_fid) + '-' + str(end_fid)

            cur_tf = optimal_data[key]
            score = match_measure(cur_tf.keys(), prev_tf.keys())

            if score < TRANSITION_THRESH and score != -1: # here is the transition, process the tfs in the past segment

                top_words = [x[0] for x in sorted(cur_tf.items(), key = operator.itemgetter(1), reverse = True)][:C_N]
                total_recog_count = sum([cur_tf[x] for x in cur_tf.keys()]) * 1.0 
                    
                
                occur = 0.0
                for w in top_words:
                    occur += cur_tf[w] 
                
                if total_recog_count > 0:
                    if occur/total_recog_count >= C_P: # this seg is specializable
                        sp_log += [(start_fid, True)]                      
                    else:
                        if len(sp_log) == 0:
                            count += 1
                            sp_log += [(start_fid, False)]
                        elif sp_log[-1][1] == True:
                            sp_log += [(start_fid, False)]
 
            prev_tf = cur_tf 
            start_fid += SLIDE_SIZE_FRAMES
        print sp_log
        with open(os.path.join( TRACE_LOG, video_name + '_' + str(UNIMPORTANTWORD_THRESH) + '.pickle' ), 'wb' )  as fh:
            pickle.dump(sp_log, fh)
        print count, count/(len(videos) * 1.0)
