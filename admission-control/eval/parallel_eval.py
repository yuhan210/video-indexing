from utils import *
from mapk import *
from ndcg import *
import os
import sys
import time
import numpy as np
import scipy.stats as stats
import operator
import pickle
import matplotlib
import matplotlib.pyplot as plt
try:
    plt.style.use('ggplot')
except:
    pass

COSSIM_THRESH = 0.05

def get_combined_tfs(tfs_dict):

    combined_tfs = {}
    # normalize
    deno = 0.0
    for d in tfs_dict:
        deno += 1
        for w in d['tf']:
            if w not in combined_tfs:
                combined_tfs[w] = 1
            else:
                combined_tfs[w] += 1

    for w in combined_tfs:
        combined_tfs[w] /= (deno * 1.0) 
    return combined_tfs


'''
# of subsampled words/# of all words
'''
def detailed_measure(all_tf, subsampled_tf):
    match_count = 0
    for w in all_tf:
        if w in subsampled_tf:
            match_count += 1
    if len(all_tf) == 0:
        return -1
    return match_count/(len(all_tf) * 1.0)
    

def combine_all_modeldicts(_vgg_data, _msr_data, _rcnn_data, _fei_data, frame_paths):

    stop_words = get_stopwords()
    wptospd = word_pref_to_stopword_pref_dict()
    convert_dict = convert_to_equal_word()

    tf_list = []

    for frame_path in frame_paths:

        frame_name = frame_path.split('/')[-1]
        rcnn_data = _rcnn_data[frame_path]
        vgg_data = _vgg_data[frame_path]
        msr_data = _msr_data[frame_path]
        fei_data = _fei_data[frame_path]
   

        # combine words
        
        rcnn_ws = []
        if len(rcnn_data) > 0:
            for rcnn_idx, word in enumerate(rcnn_data['pred']['text']):
                ## the confidence is higher than 10^(-3) and is not background
                if rcnn_data['pred']['conf'][rcnn_idx] > 0.0005 and word not in stop_words:
                    rcnn_ws += [word]

        vgg_ws = []
        if len(vgg_data) > 0:
            for vgg_idx, w in enumerate(vgg_data['pred']['text']):
                w = wptospd[w]
                if w in convert_dict:
                    w = convert_dict[w]
                prob = (-1)*vgg_data['pred']['conf'][vgg_idx]
                if w not in stop_words and prob > 0.01:
                    vgg_ws += [w]

        
        fei_ws = []
        ''' 
        if len(fei_data) > 0:
            str_list = fei_data['candidate']['text']
            for s in str_list:
                for w in s.split(' '):
                    w = inflection.singularize(w)
                    if w not in stop_words and w not in fei_ws:
                        fei_ws += [w]         
        '''
        msr_ws = [] 
        if len(msr_data) > 0:
            for msr_idx, w in enumerate(msr_data['words']['text']):
                w = inflection.singularize(w)

                prob = msr_data['words']['prob'][msr_idx]
                if w in convert_dict:
                    w = convert_dict[w]
                if w not in stop_words and len(w) != 0 and prob > -5 and msr_idx < 30:
                    msr_ws += [w]

        words = {}
        for w in rcnn_ws:
            if w not in words:
                words[w] = 1
            else:
                words[w] += 1
        for w in vgg_ws:
            if w not in words:
                words[w] = 1
            else:
                words[w] += 1
        
        for w in fei_ws:
            if w not in words:
                words[w] = 1
            else:
                words[w] += 1
    
        for w_idx, w in enumerate(msr_ws):
            if w not in words:
                words[w] = 1
            else:
                words[w] += 1

        if '' in words:
            words.pop('', None)

        tf_list += [{'frame_name': frame_name, 'tf': words}]

    return tf_list


def get_inrange_fids(start_fid, end_fid, subsampled_frames):

    in_range_fids = []
 
    for f_count, f_name in enumerate(subsampled_frames):
        fid = int(f_name.split('.')[0])
               
        if fid >= start_fid and fid < end_fid:
            in_range_fids += [fid]
 
    if len(in_range_fids) == 0:
        for f_count, f_name in enumerate(subsampled_frames):
            fid = int(f_name.split('.')[0])
            
            if f_count == len(subsampled_frames)-1:
                if fid < start_fid:
                    in_range_fids += [fid]
                    break
            elif fid < start_fid and int(subsampled_frames[f_count + 1].split('.')[0]) >= end_fid:
                    in_range_fids += [fid]
                    break

    return in_range_fids 


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

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print 'Usage: ', sys.argv[0], ' process_video_id'
        exit(11)
    
    tic = time.time()
    pvid = int(sys.argv[1])-1

    SERVER_STORAGE_FRAMES = 5 * 30 # 5 sec * 30 fps
    SLIDE_SIZE_FRAMES = 1 * 30 # 1 sec * 30 fps
    UNIMPORTANTWORD_THRESH = 0.5
    OUTPUT_FOLDER = '/home/t-yuche/admission-control/eval/greedy-results' 
    THRESH = 0.7
    sample_rate_dict = get_greedy_ave_samplerate() 
    BASELINE_SAMPLERATE = sample_rate_dict[THRESH]

    # fixed #
    GREEDY_INPUT_FOLDER = '/home/t-yuche/admission-control/window-greedy-log'
    OPTIMAL_INPUT_FOLDER = '/home/t-yuche/admission-control/eval/optimal-log/optimal-tf'
    VIDEO_LIST = '/mnt/video_list.txt'
    videos = open(VIDEO_LIST).read().split()
    
    
    uniform_scores = []
    greedy_scores = []
    for vid, video_name in enumerate(videos):

        if vid != pvid:
            continue
        print video_name
 
       
        # safe rerun
        outgreedypath = os.path.join(OUTPUT_FOLDER, video_name + '_greedy_' + str(THRESH) + '_' + str(BASELINE_SAMPLERATE) + '.pickle')
        outuniformpath = os.path.join(OUTPUT_FOLDER, video_name + '_uniform_' + str(THRESH) + '_' + str(BASELINE_SAMPLERATE) + '.pickle')
        #if os.path.exists(outgreedypath) and os.path.exists(outuniformpath):
        #    continue    
   

        # load all models 
        rcnn_dict, vgg_dict, fei_caption_dict, msr_cap_dict, dummy = load_all_modules_dict(video_name)
        frame_paths = sorted(vgg_dict, key = lambda x: int(x.split('/')[-1].split('.')[0])) 

        # load greedy subsampled frames
        greedy_gt_path = os.path.join(GREEDY_INPUT_FOLDER, video_name +  '_' + str(THRESH) + '_gtframe.pickle')
        with open(greedy_gt_path) as gt_fh:
            selected_frame_obj = pickle.load(gt_fh)
            greedy_frames = [ str(x) + '.jpg' for x in selected_frame_obj['picked_f']]
            video_len_f = selected_frame_obj['total_frame']
            subsampled_rate = selected_frame_obj['picked_rate']

        # baseline
        uniform_frames = naive_subsample_frames(os.listdir(os.path.join('/mnt/frames', video_name)), BASELINE_SAMPLERATE) 

        # load optimal tfs
        with open(os.path.join(OPTIMAL_INPUT_FOLDER, video_name + '_' + str(UNIMPORTANTWORD_THRESH) + '.pickle')) as fh:
            optimal_data = pickle.load(fh)

        # sliding window   
        video_start_fid = 0
        video_end_fid = 0
        while True:

            if video_start_fid > video_len_f - 1:
                break  
            video_end_fid = min(video_start_fid + SERVER_STORAGE_FRAMES, video_len_f)

            key = str(video_start_fid) + '-' + str(video_end_fid)
 
            ''' Optimal '''
            optimal_tf = optimal_data[key]
            #print optimal_tf
    
            ''' Uniform '''
            uni_range_fids = get_inrange_fids(video_start_fid, video_end_fid, uniform_frames)
            uniform_inrange_frames = [os.path.join('/mnt/frames', video_name, str(x) + '.jpg') for x in uni_range_fids] 
            uniform_tf_list = combine_all_modeldicts(vgg_dict, msr_cap_dict, rcnn_dict, fei_caption_dict, uniform_inrange_frames)
            uniform_tf = get_combined_tfs(uniform_tf_list)
            uniform_score = detailed_measure(optimal_tf, uniform_tf) 
           

            ''' Greedy '''
            greedy_range_fids = get_inrange_fids(video_start_fid, video_end_fid, greedy_frames)
            greedy_inrange_frames = [os.path.join('/mnt/frames', video_name, str(x) + '.jpg') for x in greedy_range_fids] 
            greedy_tf_list = combine_all_modeldicts(vgg_dict, msr_cap_dict, rcnn_dict, fei_caption_dict, greedy_inrange_frames)
            greedy_tf = get_combined_tfs(greedy_tf_list)
            greedy_score = detailed_measure(optimal_tf, greedy_tf) 

            #print greedy_tf 
            #print uniform_score, greedy_score

            uniform_scores += [uniform_score]
            greedy_scores += [greedy_score]
            video_start_fid +=  SLIDE_SIZE_FRAMES 

        #plot_scores(video_name, start_fids, greedy_scores, uniform_scores, selected_frame_obj['picked_f'], [int(x.split('.')[0]) for x in uniform_frames], subsampled_rate, BASELINE_SAMPLERATE)
        with open(outgreedypath, 'wb') as fh:
            pickle.dump(greedy_scores, fh)
        with open(outuniformpath, 'wb') as fh:
            pickle.dump(uniform_scores, fh)
        break
    toc = time.time()
    print 'Exec time:', toc-tic
    print 'video_name:', video_name
    print 'greedy_score:', np.mean(greedy_scores), '  uniform score:', np.mean(uniform_scores)
