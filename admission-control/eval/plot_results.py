import os
import math
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
try:
    plt.style.use('ggplot')
except:
    pass


def get_greedy_ave_samplerate(file_name = '/home/t-yuche/admission-control/tools/greedy_sample_rates'):

    lines = open(file_name).readlines()
    sample_rates = {}

    for line in lines:
        line = line.strip()
        segs = [float(x) for x in line.split()]
        thresh = segs[0]
        sample_rate = segs[1]
        sample_rates[thresh]  = sample_rate

    return sample_rates

def getAllVideosRecall(greedy_thresh, sample_rate, RESULT_FOLDER = '/home/t-yuche/admission-control/eval/greedy-results'):

    VIDEO_LIST = '/mnt/video_list.txt'
    videos = open(VIDEO_LIST).read().split()
   
    greedy_scores = []
    uniform_scores = [] 
    for video_name in videos:
    
        g, u = getPerVideoRecall(video_name, greedy_thresh, sample_rate)
        greedy_scores += g 
        uniform_scores += u

    return greedy_scores, uniform_scores 

def getPerVideoRecall(video_name, greedy_thresh, sample_rate, RESULT_FOLDER= '/home/t-yuche/admission-control/eval/greedy-results' ):
    greedy_fname = os.path.join(RESULT_FOLDER, video_name + '_greedy_' + str(greedy_thresh) + '_' + str(sample_rate) + '.pickle')
    uniform_fname = os.path.join(RESULT_FOLDER, video_name + '_uniform_' + str(greedy_thresh) + '_' + str(sample_rate) + '.pickle')

    with open(greedy_fname) as fh:
        greedy_scores = pickle.load(fh)
    with open(uniform_fname) as fh:
        uniform_scores = pickle.load(fh)
    
    return greedy_scores, uniform_scores

if __name__ == "__main__":


    RESULT_FOLDER = '/home/t-yuche/admission-control/eval/greedy-results' 
    VIDEO_LIST = '/mnt/video_list.txt'
    videos = open(VIDEO_LIST).read().split()
    sample_rates_dict = get_greedy_ave_samplerate()
    #THRESHS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]    
    THRESHS = [0.8]    

    uniform_rates = []
    uniform_acc = []
    uniform_accerr = []

    greedy_rates = []
    greedy_acc = []
    greedy_accerr = []     

    for thresh in THRESHS:
        g, u = getAllVideosRecall(thresh, sample_rates_dict[thresh])   
        print 'Greedy:', np.mean(g), np.std(g, ddof = 1) 
        print 'Uniform:', np.mean(u), np.std(u, ddof = 1)

        uniform_rates += [sample_rates_dict[thresh]]          
        greedy_rates += [sample_rates_dict[thresh]] 
        
        uniform_acc += [np.mean(u)]
        greedy_acc += [np.mean(g)]
        
        
        uniform_accerr += [np.std(u, ddof = 1)/math.sqrt(len(u))]
        greedy_accerr += [np.std(g, ddof = 1)/math.sqrt(len(g))]


    uniform_rates += [1]
    greedy_rates += [1]
    
    uniform_acc += [1]
    greedy_acc += [1]
    
    uniform_accerr += [0]
    greedy_accerr += [0]


    plt.figure()
    plt.errorbar(uniform_rates, uniform_acc, yerr=uniform_accerr, color = 'b', fmt = '-o', label = 'Uniform')
    plt.xscale('log')
    plt.errorbar(greedy_rates, greedy_acc, yerr=greedy_accerr, color = 'r', fmt = '-x', label = 'Greedy')
    plt.xscale('log')
    plt.ylim([0, 1])
    plt.xlabel('Subsample rate')
    plt.ylabel('Average accuracy')
    plt.legend()
    plt.show()

