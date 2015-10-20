from nltk.corpus import stopwords
import scipy.spatial.distance
import numpy as np
import operator
import pickle
import shutil
import math
import time
import sys
from utils import *
from nlp import *
import matplotlib
import matplotlib.pyplot as plt
try:
    plt.style.use('ggplot')
except:
    pass
import cv2

def remove_unimportantwords(tf):
    new_tf = {}
    for w in tf:
        if tf[w] > 0.5:
            new_tf[w] = tf[w]

    return new_tf

def get_inrange_fids(start_fid, end_fid, subsampled_fids):

    in_range_fids = []
    for f_count, fid in enumerate(subsampled_fids):
               
        if fid >= start_fid and fid < end_fid:
            in_range_fids += [fid]
 
    if len(in_range_fids) == 0:
        for f_count, fid in enumerate(subsampled_fids):
            
            if f_count == len(subsampled_fids)-1:
                if fid < start_fid:
                    in_range_fids += [fid]
                    break
            elif fid < start_fid and subsampled_fids[f_count + 1] >= end_fid:
                    in_range_fids += [fid]
                    break

    return in_range_fids 



def combine_all_modeldicts(_vgg_data, _msr_data, _rcnn_data, _fei_data, frame_paths):

    stop_words = get_stopwords()
    wptospd = word_pref_to_stopword_pref_dict()
    convert_dict = convert_to_equal_word()

    tf_list = {}
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

        tf_list[frame_name] = words
    
    return tf_list



def get_combined_tfs(tfs_dict):

    combined_tfs = {}
    # normalize
    deno = len(tfs_dict)
    for frame_name in tfs_dict:
        tf = tfs_dict[frame_name]
        for w in tf:
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
    


def subsample_tf_list(selected_frames, all_tf_list):
   
    tfs = []
    for fid, frame_name in enumerate(selected_frames):
        #print fid, '/', len(selected_frames)
        frame_id = int(frame_name.split('.')[0])
        
        tf = filter(lambda x: int(x['frame_name'].split('.')[0]) == frame_id, all_tf_list)
        tfs += [tf[0]]  

    return tfs


def framediff_select(video_name, frame_names):

    MOVEMENT_PERCENTAGE = 0.5
    prev_framename = None
    retained_frames = []

    for frame_name in frame_names:
        if prev_framename == None:
            retained_frames += [frame_name]
            prev_framename = frame_name
            continue
                
        # load prev_frame and cur_frame
        cur_frame = cv2.imread(os.path.join('/mnt/frames/', video_name, frame_name), 0)       
        prev_frame = cv2.imread(os.path.join('/mnt/frames/', video_name, prev_framename), 0)

        diff = cv2.absdiff(cur_frame, prev_frame)
        cv2.imshow('diff', diff)
        ret, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
        cv2.imshow('thresh', thresh)
        cv2.waitKey(1000)
        frame_diff = cv2.countNonZero(thresh)

        if frame_diff >= (MOVEMENT_PERCENTAGE * cur_frame.shape[0] * cur_frame.shape[1]):
            retained_frames += [frame_name]
            prev_framename = frame_name

    return retained_frames


if __name__ == "__main__":
    
    SERVER_STORAGE_FRAMES = 5 * 30 # 5 sec * 30 fps
    SLIDE_SIZE_FRAMES = 1 * 30 # 1 sec * 30 fps
    THRESH = 0.9
     
    VIDEO_LIST = '/mnt/video_list.txt'
    videos = open(VIDEO_LIST).read().split()

    for vid, video_name in enumerate(videos):
 
        if not os.path.exists(os.path.join('/mnt/tags/rcnn-info-all', video_name + '_rcnnrecog.json')) or not os.path.exists(os.path.join('/mnt/tags/vgg-classify-all', video_name + '_recog.json')) or not os.path.exists(os.path.join('/mnt/tags/msr-caption-all', video_name + '_msrcap.json')) or not os.path.exists(os.path.join('/mnt/tags/fei-caption-all', video_name + '_5_caption.json')):
            continue
      
        print video_name
        outfilepath = os.path.join('window-greedy-log', video_name + '_' + str(THRESH)  + '_gtframe.pickle')

        if os.path.exists(outfilepath):
            continue
        
        # load tags from all DNN modules 
        rcnn_dict, vgg_dict, fei_caption_dict, msr_cap_dict, dummy = load_all_modules_dict(video_name)
     
        _turker_data, dummy = load_video_processed_turker(video_name)
 
        # compose video term freq (a list of dicts)
        frame_names = os.listdir(os.path.join('/mnt/frames', video_name))
        frame_names = sorted(frame_names, key= lambda x: int(x.split('.')[0]))
        frame_paths = [os.path.join('/mnt/frames', video_name, x) for x in frame_names]
        all_tfs_list = combine_all_modeldicts(vgg_dict, msr_cap_dict, rcnn_dict, fei_caption_dict, frame_paths)
        video_len = len(frame_names) 

        start_fid = 0
        end_fid = 0
        selected_fids = [0] 
        scores = []
        start_fids = []
        while True:
            
            end_fid = min(start_fid + SERVER_STORAGE_FRAMES, video_len)
            #print 'start:', start_fid, ' end:', end_fid
            if start_fid > video_len - 1:
                break

            # optimal tf list
            inrange_optimal_tfs_list = {}
            for fname in all_tfs_list:
                fid = int(fname.split('.')[0])
                if fid >= start_fid and fid < end_fid:
                    inrange_optimal_tfs_list[fname] = all_tfs_list[fname]

            optimal_tf = get_combined_tfs(inrange_optimal_tfs_list)
            #print '\nOriginal optimal_tf:', optimal_tf
            optimal_tf = remove_unimportantwords(optimal_tf)
            #print 'Filtered optimal_tf:', optimal_tf

            greedy_inrange_fids = get_inrange_fids(start_fid, end_fid, selected_fids)
            greedy_inrange_frames = [os.path.join('/mnt/frames', video_name, str(x) + '.jpg') for x in greedy_inrange_fids] 
            greedy_tf_list = combine_all_modeldicts(vgg_dict, msr_cap_dict, rcnn_dict, fei_caption_dict, greedy_inrange_frames)
            greedy_tf = get_combined_tfs(greedy_tf_list)
            #print 'Greedy_tf:', greedy_tf 
            cur_score = detailed_measure(optimal_tf, greedy_tf)
            #print 'Cur score:', cur_score
            scores += [cur_score]
            start_fids += [start_fid]

            if cur_score != -1 and cur_score < THRESH:

                # expire old fids                
                greedy_inrange_fids = filter(lambda x: x >= start_fid, greedy_inrange_fids)
                if len(greedy_inrange_fids) > 0: 
                    search_start_fid = max(greedy_inrange_fids)
                else:
                    search_start_fid = start_fid

                #print 'After expiring old fids:', greedy_inrange_fids
                # search for the best aggregated fids
                score_logs = {}
                max_recall = -1
                max_recall_cand = []
                #print start_fid, end_fid
                for i in xrange(search_start_fid + 1, end_fid):
                    # maximize precision
                    tmp_inrange_fids = greedy_inrange_fids
                    tmp_inrange_fids += [i]

                    # compute new score 
                    tmp_inrange_frames = [str(x) + '.jpg' for x in tmp_inrange_fids] 
                
                    tmp_tf_list = {}
                    for fname in tmp_inrange_frames:
                        tmp_tf_list[fname] = all_tfs_list[fname]

                    tmp_tf = get_combined_tfs(tmp_tf_list)
                    tmp_score = detailed_measure(optimal_tf, tmp_tf)
                    if tmp_score > max_recall:
                        max_recall_cand = [i]
                        max_recall = tmp_score
                    elif tmp_score == max_recall:
                        max_recall_cand += [i]


                    #print 'Adding fid:', i, 'recall:', tmp_score, ' len:', len(all_tfs_list[tmp_inrange_frames[-1]])
                    score_logs[i] = [tmp_score, len(all_tfs_list[tmp_inrange_frames[-1]])] 

                #print score_logs
                # select the best frame
                #print 'Max recall cand:', max_recall_cand
                max_coverage = -1
                max_coverage_cand = []
                for candidate_fid in max_recall_cand:
                    if score_logs[candidate_fid][1] > max_coverage:
                        max_coverage_cand = [candidate_fid]
                        max_coverage = score_logs[candidate_fid][1] 
                    elif score_logs[candidate_fid][1] == max_coverage:
                        max_coverage_cand += [candidate_fid] 
 
                # select
                if len(max_coverage_cand) > 0:
                    select_fid = max(max_coverage_cand)
                    selected_fids += [select_fid]
                    scores[-1] = max_recall
                #print 'Select to add fid:', max(max_coverage_cand)
             
            elif cur_score == -1:
                # this chunk is bad .. currently - ignore, till a good frame appears
                selected_fids = selected_fids                

            #print selected_fids 
            #print scores
            # update selected fids
            start_fid += SLIDE_SIZE_FRAMES
             
              
        print len(selected_fids) 
        
        with open(outfilepath, 'wb') as gt_fh:
            pickle.dump({'picked_f': selected_fids, 'total_frame': len(all_tfs_list), 'picked_rate': len(selected_fids)/(len(all_tfs_list) * 1.0)}, gt_fh)
   
        '''
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.set_title('# subsampled distinct word/# nonsubsampled distinct word (bigger -> more similar)')
        
        plt.text(video_len/2, THRESH - 0.2, 'Portion of selected frames:' + str(len(selected_fids)/(len(all_tfs_list) * 1.0)),fontsize = 20)
        #print start_fids, scores
        #print len(start_fids), len(scores)
        plt.plot(start_fids, scores, '-x')
        plt.ylim([0, 1.02])
        plt.scatter(selected_fids, [0.1 for x in selected_fids], marker = (5,1))
        plt.plot([0, len(all_tfs_list)], [THRESH, THRESH], 'k-')
        ax.set_xlabel('Frame ID')
        ax.set_ylabel('# subsampled distinct word/# nonsubsampled distinct word')
        ax.set_ylim([0,1]) 
        plt.show()
 
        break 
        '''       
