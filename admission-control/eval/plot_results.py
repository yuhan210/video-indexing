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

def getAllVideosUniformRecall(sample_rate, RESULT_FOLDER = '/home/t-yuche/admission-control/eval/greedy-results'):

    VIDEO_LIST = '/mnt/video_list.txt'
    videos = open(VIDEO_LIST).read().split()
   
    uniform_scores = [] 
    for video_name in videos:
    
        u = getPerVideoUniformRecall(video_name, sample_rate)
        uniform_scores += u

    return uniform_scores 

def getPerVideoUniformRecall(video_name, sample_rate, RESULT_FOLDER= '/home/t-yuche/admission-control/eval/greedy-results' ):
    uniform_fname = os.path.join(RESULT_FOLDER, video_name + '_uniform_' + str(0) + '_' + str(sample_rate) + '.pickle')

    with open(uniform_fname) as fh:
        uniform_scores = pickle.load(fh)
    
    if -1 in uniform_scores: uniform_scores.remove(-1)
    return uniform_scores

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
    
    if -1 in greedy_scores: greedy_scores.remove(-1)
    if -1 in uniform_scores: uniform_scores.remove(-1)
    return greedy_scores, uniform_scores

if __name__ == "__main__":


    RESULT_FOLDER = '/home/t-yuche/admission-control/eval/greedy-results' 
    VIDEO_LIST = '/mnt/video_list.txt'
    videos = open(VIDEO_LIST).read().split()
    sample_rates_dict = get_greedy_ave_samplerate()
    THRESHS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]    
    #BASELINE_SAMPLERATE = [0.001] 
    BASELINE_SAMPLERATE = [] 
    #BASELINE_SAMPLERATE = [0.01, 0.1] 

    greedy_results = {}
    uniform_results = {}
    for thresh in THRESHS:
        g, u = getAllVideosRecall(thresh, sample_rates_dict[thresh])   
        print 'Greedy:', np.mean(g), np.std(g, ddof = 1)/math.sqrt(len(g)) 
        print 'Uniform:', np.mean(u), np.std(u, ddof = 1)/math.sqrt(len(u))

        
        greedy_results[sample_rates_dict[thresh]]  = (np.mean(g), np.std(g, ddof = 1)/math.sqrt(len(g)))
        uniform_results[sample_rates_dict[thresh]]  = (np.mean(u), np.std(u, ddof = 1)/math.sqrt(len(u)))
        '''
        uniform_accerr += [np.std(u, ddof = 1)/2.]
        greedy_accerr += [np.std(g, ddof = 1)/2.]

        '''

    for b_samplerate in BASELINE_SAMPLERATE:
        u = getAllVideosUniformRecall(b_samplerate)   
        uniform_results[b_samplerate]  = (np.mean(u), np.std(u, ddof = 1)/math.sqrt(len(u)))
    
    #greedy_results[0.01]  = (1, 0)
    #uniform_results[1]  = (1, 0)


    plt.figure()
    plt.errorbar(sorted(uniform_results.keys()), [uniform_results[x][0]*100 for x in sorted(uniform_results.keys())], yerr=[uniform_results[x][1] for x in sorted(uniform_results.keys())], color = 'b', fmt = '--x', label = 'Uniform')
    #plt.xscale('log')
    plt.errorbar(sorted(greedy_results.keys()), [greedy_results[x][0]*100 for x in sorted(greedy_results.keys())], yerr=[greedy_results[x][1] for x in sorted(greedy_results.keys())], color = 'r', fmt = '-o', label = 'Greedy')
    #plt.xscale('log')
    plt.ylim([0, 100])
    plt.xlim([0.001, 0.0095])
    plt.xlabel('Subsample Rate')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc = 4)
    #plt.show()
    plt.savefig('acc_vs_samplerate_greedy.pdf', bbox_inches = 'tight')
