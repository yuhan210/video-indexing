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
font = {
        'size'   : 24,
    }
text = {'color': 'black'}
line = {'linewidth': 3}
matplotlib.rc('font', **font)
matplotlib.rc('text', **text)
matplotlib.rc('lines', **line)



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

    OPTIMAL_INPUT_FOLDER = '/home/t-yuche/admission-control/eval/optimal-log/optimal-tf'
    VIDEO_LIST = '/mnt/video_list.txt'
    SERVER_STORAGE_FRAMES = 5 * 30 # 5 sec * 30 fps
    SLIDE_SIZE_FRAMES = 1 * 30 # 1 sec * 30 fps
    UNIMPORTANTWORD_THRESH = 0.5

    TRANSITION_THRESH = 0.2
      
    videos = open(VIDEO_LIST).read().split()

    cdf_value = {}
    for i in xrange(1,400):
        cdf_value[i] = []

    for vid, video_name in enumerate(videos):
        print video_name
        n_frames = get_video_frame_num(video_name)
        
        scatter_data = [] 
        with open(os.path.join(OPTIMAL_INPUT_FOLDER, video_name + '_' + str(UNIMPORTANTWORD_THRESH) + '.pickle')) as fh:
            optimal_data = pickle.load(fh)
     
        all_words = load_all_words()

        start_fid = 0
        prev_tf = {} 
        prev_cum_tf = {}
        prev_tran_start = 0
        all_prev_tfs = []
        while True:
            
            if start_fid > n_frames - 1:
                break
        
            end_fid = min(start_fid + SERVER_STORAGE_FRAMES, n_frames) 
            key = str(start_fid) + '-' + str(end_fid)

            cur_tf = optimal_data[key]   
            score = match_measure(cur_tf.keys(), prev_tf.keys())

            if score < TRANSITION_THRESH and start_fid > 0 and score != -1: # here is the transition, process the tfs in the past segment


                top_words = [x[0] for x in sorted(prev_cum_tf.items(), key = operator.itemgetter(1), reverse = True)]
                for wid in xrange(len(top_words)):
                    cum_occur = 0
                    for i in xrange(wid + 1):
                        word = top_words[i]
                        cum_occur += prev_cum_tf[word]
                    prec = cum_occur/(sum([prev_cum_tf[x] for x in prev_cum_tf.keys()]) * 1.0)

                    cdf_value[wid + 1] += [prec]



                prev_tran_start = start_fid
                prev_cum_tf = {}
                all_prev_tfs = []
          
            all_prev_tfs += [cur_tf] 
            for w in cur_tf.keys():
                if w not in prev_cum_tf:
                    prev_cum_tf[w] = 1
                else:
                    prev_cum_tf[w] += 1
 
            prev_tf = cur_tf 
  
            start_fid += SLIDE_SIZE_FRAMES

    x_value = {}
    stop_value = 65
    for i in xrange(1,400):
        m = np.mean(cdf_value[i]) 
        std = np.std(cdf_value[i])
        x_value[i] = (m, std)

    plt.figure()   
    plt.xlabel('Top-k word') 
    plt.ylabel('Occurrence Percentage (%)') 
    xs = range(1, 66)    
    plt.errorbar(xs, [x_value[x][0] * 100 for x in xs], yerr=[x_value[x][1] * 100 for x in xs], capsize = 3)
    plt.ylim([0, 100])
    plt.savefig('skew_cdf.pdf', bbox_inches = 'tight')
