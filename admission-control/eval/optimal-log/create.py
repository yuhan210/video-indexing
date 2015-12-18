from utils import *
from mapk import *
from ndcg import *
import os
import sys
import time
import numpy as np
import scipy.stats as stats
import operator
from vision import *
import pickle
import matplotlib
import matplotlib.pyplot as plt
try:
    plt.style.use('ggplot')
except:
    pass


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


def remove_unimportantwords(tf, thresh = 0.5):
    new_tf = {}
    for w in tf:
        if tf[w] > thresh:
            new_tf[w] = tf[w]

    return new_tf


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
        #fei_data = _fei_data[frame_path]
   

        # combine words
        '''        
        rcnn_ws = []
        if len(rcnn_data) > 0:
            for rcnn_idx, word in enumerate(rcnn_data['pred']['text']):
                ## the confidence is higher than 10^(-3) and is not background
                if rcnn_data['pred']['conf'][rcnn_idx] > 0.0005 and word not in stop_words:
                    rcnn_ws += [word]
        '''
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
        '''
        for w in rcnn_ws:
            if w not in words:
                words[w] = 1
            else:
                words[w] += 1
        '''
        for w in vgg_ws:
            if w not in words:
                words[w] = 1
            else:
                words[w] += 1
        ''' 
        for w in fei_ws:
            if w not in words:
                words[w] = 1
            else:
                words[w] += 1
        '''
        for w_idx, w in enumerate(msr_ws):
            if w not in words:
                words[w] = 1
            else:
                words[w] += 1

        if '' in words:
            words.pop('', None)

        tf_list += [{'frame_name': frame_name, 'tf': words}]

    return tf_list


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print 'Usage:', sys.argv[0], ' processed_vid'
        exit(11)

    pvid = int(sys.argv[1])-1
    SERVER_STORAGE_FRAMES = 5 * 30 # 5 sec * 30 fps
    SLIDE_SIZE_FRAMES = 1 * 30 # 1 sec * 30 fps
    UNIMPORTANTWORD_THRESH = 0.5
    

    VIDEO_LIST = '/mnt/video_list.txt'
    videos = open(VIDEO_LIST).read().split()

    for vid, video_name in enumerate(videos):
        if vid != pvid:
            continue
        print pvid, video_name      
    
        outputpath = '/home/t-yuche/admission-control/eval/optimal-log/optimal-tf/' + video_name + '_' + str(UNIMPORTANTWORD_THRESH) + '.pickle'
        '''
        if os.path.exists(outputpath):
            print 'exists'
            continue
        '''
        rcnn_dict, vgg_dict, fei_caption_dict, msr_cap_dict, dummy = load_all_modules_dict_local(video_name)
        video_len_f = len(vgg_dict)
        # sliding window   
        video_start_fid = 0
        video_end_fid = 0
        optimal_tf_dict = {}
        cvdata = getCVInfoFromLog(video_name)

        while True:

            if video_start_fid > video_len_f - 1:
                break  
            video_end_fid = min(video_start_fid + SERVER_STORAGE_FRAMES, video_len_f)
            #print 'start:', video_start_fid, ' end:', video_end_fid
            
   
            ''' Optimal '''
            optimal_frames = [os.path.join('/mnt/frames', video_name, str(x) + '.jpg') for x in range(video_start_fid, video_end_fid)]
            # get illumination
                
            illus = []
            for cv_fid in xrange(video_start_fid, video_end_fid):
                framename = str(cv_fid) + '.jpg'
                cv = cvdata[framename]
                illus += [cv['illu'][0]]
            if np.mean(illus) < 10:
                optimal_tf = {}
            else: 
                optimal_tf_list = combine_all_modeldicts(vgg_dict, msr_cap_dict, rcnn_dict, fei_caption_dict, optimal_frames)
                optimal_tf = get_combined_tfs(optimal_tf_list)
                optimal_tf = remove_unimportantwords(optimal_tf,  UNIMPORTANTWORD_THRESH)
            key = str(video_start_fid) + '-' + str(video_end_fid)
            optimal_tf_dict[key] = optimal_tf

            video_start_fid +=  SLIDE_SIZE_FRAMES 

        with open(outputpath, 'wb') as fh:
            pickle.dump(optimal_tf_dict, fh)
