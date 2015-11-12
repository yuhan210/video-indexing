from utils import *
from mapk import *
from ndcg import *
import os
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


def remove_unimportantwords(tf):
    new_tf = {}
    for w in tf:
        if tf[w] > 0.5:
            new_tf[w] = tf[w]

    return new_tf

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


def combine_all_models(video_name, _vgg_data, _msr_data, _rcnn_data, _fei_data):

    stop_words = get_stopwords()
    wptospd = word_pref_to_stopword_pref_dict()
    convert_dict = convert_to_equal_word()

    tf_list = []
    assert(len(_vgg_data) == len(_msr_data))
    assert(len(_rcnn_data) == len(_fei_data))
    assert(len(_vgg_data) == len(_fei_data))

    for fid in xrange(len(_vgg_data)):

        rcnn_data = _rcnn_data[fid]
        vgg_data = _vgg_data[fid]
        msr_data = _msr_data[fid]
        fei_data = _fei_data[fid]
   
        frame_name = rcnn_data['image_path'].split('/')[-1]
        assert(rcnn_data['image_path'] == vgg_data['img_path'])
        assert(rcnn_data['image_path'] == msr_data['img_path'])
        assert(rcnn_data['image_path'] == fei_data['img_path'])

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
                prob = vgg_data['pred']['conf'][vgg_idx]
            
                if w not in stop_words:
                    vgg_ws += [w]
    
        fei_ws = [] 
        if len(fei_data) > 0:
            str_list = fei_data['candidate']['text']
            for s in str_list:
                for w in s.split(' '):
                    w = inflection.singularize(w)
                    if w not in stop_words and w not in fei_ws:
                        fei_ws += [w]         

        msr_ws = [] 
        if len(msr_data) > 0:
            for msr_idx, w in enumerate(msr_data['words']['text']):
                w = inflection.singularize(w)
                prob = msr_data['words']['prob'][msr_idx]
                if w not in stop_words:
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

def load():
    
    VIDEO_LIST = '/mnt/video_list.txt'
    videos = open(VIDEO_LIST).read().split()
    
    vgg = {}
    fei_cap = {}
    msr_cap = {}
    rcnn = {}
    turker_labels = {}
    for vid, video_name in enumerate(videos):
        _vgg_data = load_video_recog('/mnt/tags/vgg-classify-all', video_name)
        _fei_caption_data = load_video_caption('/mnt/tags/fei-caption-all', video_name)
        _msr_cap_data = load_video_msr_caption('/mnt/tags/msr-caption-all', video_name)
        _rcnn_data = load_video_rcnn('/mnt/tags/rcnn-info-all', video_name)
        _turker_data = load_video_processed_turker(video_name)       
 
        vgg[video_name] = _vgg_data[0]
        fei_cap[video_name] = _fei_caption_data[0]
        msr_cap[video_name] = _msr_cap_data[0]
        rcnn[video_name] = _rcnn_data[0]
        turker_labels[video_name] = _turker_data[0]

        if vid == 30:
            break
    return vgg, fei_cap, msr_cap, rcnn, turker_labels


def rank(query, video_tfs):
   
    # 
    query_dict = {}
    query_dict[query] = 1

    video_scores = {}
    for video_d in video_tfs:
        video_name = video_d['video_name']
        video_tf = video_d['tf']
        cos_sim = cos_similarty(query_dict, video_tf)  
        video_scores[video_name] = cos_sim

    ranked_video = sorted(video_scores.items(), key = operator.itemgetter(1), reverse=True)

    return ranked_video


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


def get_turker_subsampled_tf(video_name, turker_data, in_range_fids):

    turker_tf = {} 
    for fid in in_range_fids:
        for label in filter(lambda x: int(x['frame_name'].split('.')[0]) == fid, turker_data)[0]['gt_labels']:
            if label not in turker_tf:
                turker_tf[label] = 1
            else:
                turker_tf[label] += 1
       
    return turker_tf 

def get_subsampled_tf_from_dict(vgg_dict, msr_dict, rcnn_dict, fei_dict, in_range_videopaths):


    s_tf_list = combine_all_models(dummy, s_vgg_data, s_msr_data, s_rcnn_data, s_fei_data)
    s_tf = get_combined_tfs(s_tf_list)
    return s_tf

def get_subsampled_tf(video_name, vgg_data, msr_data, rcnn_data, fei_data, in_range_fids):

    s_vgg_data = []
    s_fei_data = []
    s_msr_data = []
    s_rcnn_data = []
    
    for fid in in_range_fids:

        s_vgg_data += filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == fid , vgg_data)
        s_fei_data += filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == fid , fei_data)
        s_msr_data += filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == fid , msr_data)
        s_rcnn_data += filter(lambda x: int(x['image_path'].split('/')[-1].split('.')[0]) == fid, rcnn_data)

    s_tf_list = combine_all_models(video_name, s_vgg_data, s_msr_data, s_rcnn_data, s_fei_data)
    s_tf = get_combined_tfs(s_tf_list)

    return s_tf


def reindexing(gt_video_rank, test_video_rank, score_thresh):
    """
    Indexing ranked video list using ground-truth video ranking
    """ 

    gt_rank = []
    k = 0
    for tid, tup in enumerate(gt_video_rank):
        video_name = tup[0]
        score = tup[1]
        if score > score_thresh:
            k += 1
        gt_rank += [video_name] 
     
    test_rank = [] 
    for tvid, tup in enumerate(test_video_rank):
        t_video_name = tup[0]
        t_score = tup[1]
        test_rank += [gt_rank.index(t_video_name)]

    return test_rank, k


# TODO: only consider score > 0
def kendall_correlation(gt_video_rank, test_video_rank):
  
    assert(len(gt_video_rank) == len(test_video_rank))
    tau, p_value = stats.kendalltau(range(len(gt_video_rank)), test_video_rank)
    
    return tau, p_value 
      

def plot_scores(video_name, start_fids, greedy_scores, uniform_scores, greedy_fids, uniform_fids, greedy_sample_rate, uniform_sample_rate):
    plt.figure(figsize=(13,11))
    plt.plot(start_fids, greedy_scores, 'ro-', label = 'Greedy Accuracy')
    plt.plot(start_fids, uniform_scores, 'b-o', label = 'Uniform Accuracy')
    plt.scatter(greedy_fids, [0.1 for x in xrange(len(greedy_fids))] , color = 'red', label = 'Greedy processed frame ID')
    plt.scatter(uniform_fids, [0.05 for x in xrange(len(uniform_fids))] , color = 'blue', label = 'Uniform processed frame ID')
    plt.text(10, 1.1, 'Greedy subsample rate:' + str(greedy_sample_rate) + '\n Uniform subsample rate:' + str(uniform_sample_rate),fontsize = 15)

    plt.title(video_name)
    plt.legend(loc = 4)
    plt.savefig('/home/t-yuche/admission-control/eval/anec-figs/thresh_08/' + video_name + '.png', dpi = 100)
    #plt.show()
    return 

def plot_rank_debug(gt_rank, gt_tfs, test_tfs, video_rank, query_str, test_scheme_name = "groundtruth"):
    
    fig = plt.figure()
    fig.suptitle(test_scheme_name + ' query:' + query_str)
    for rank, tup in enumerate(gt_rank):
        video_name = tup[0]
        score = tup[1] 
       
        # get test tf
        tf = filter(lambda x: x['video_name'] == video_name, test_tfs)[0]['tf']
        deno = sum([tf[x] for x in tf]) * 1.0
        # get gt tf
        gt_tf = filter(lambda x:x['video_name'] == video_name, gt_tfs)[0]['tf']
        gt_deno = sum([gt_tf[x] for x in gt_tf]) * 1.0

        gt_x = gt_tf.keys()         
        # align test tf based on gt tf
        x = []
        y = []
        gt_y = []
        
        query_point = (-1,-1) 
        gt_query_point = (-1,-1) 
        for xidx, key in enumerate(gt_x):
            gt_y += [gt_tf[key]/gt_deno]

            if key in tf:
                y += [tf[key]/deno] 
                if key == query_str.split('-')[0]:
                    query_point = (xidx, tf[key]/deno) 
                    gt_query_point = (xidx, gt_tf[key]/gt_deno) 
            else:
                y += [0]

        for key in tf:
            if key not in x:
                x += [key]
                y += [tf[key]/deno]

        # get test video rank
        t_rank = -1
        t_score = -1
        for trank, ttup in enumerate(video_rank):
            if ttup[0] == video_name:
                t_rank = trank
                t_score = ttup[1]
                break 

        ax = plt.subplot(len(gt_rank)/5,  5, rank+1)
        ax.plot( range(len(x)), y, color = 'blue', alpha = 0.7, label = test_scheme_name ) 
        ax.plot( range(len(gt_x)), gt_y, color = 'red', alpha = 0.7, label = "groundtruth")
        if query_point[0] != -1:
            ax.scatter(query_point[0], query_point[1], color = 'blue', marker = (5,1)) 
        if gt_query_point[0] != -1:
            ax.scatter(gt_query_point[0], gt_query_point[1], color = 'red', marker = (5,1)) 
        ax.set_title(video_name[3:13] + ' r:' + str(t_rank) + ' s:' + str(t_score) )
        ax.legend()
    plt.show()



if __name__ == "__main__":

    SERVER_STORAGE_FRAMES = 5 * 30 # 5 sec * 30 fps
    SLIDE_SIZE_FRAMES = 1 * 30 # 1 sec * 30 fps
    BASELINE_SAMPLERATE = 0.0081
    GREEDY_FOLDER = '/home/t-yuche/admission-control/eval/greedy-results' 
    THRESH = 0.9 

    GREEDY_INPUT_FOLDER = '/home/t-yuche/admission-control/window-greedy-log'
    VIDEO_LIST = '/mnt/video_list.txt'
    videos = open(VIDEO_LIST).read().split()

    for vid, video_name in enumerate(videos):
        print video_name      
       
        outgreedypath = os.path.join(GREEDY_FOLDER, video_name + '_greedy_' + str(THRESH) + '_' + str(BASELINE_SAMPLERATE) + '.pickle')
        outuniformpath = os.path.join(GREEDY_FOLDER, video_name + '_uniform_' + str(THRESH) + '_' + str(BASELINE_SAMPLERATE) + '.pickle')

        if os.path.exists(outgreedypath) and os.path.exists(outuniformpath):
            continue    
    
        tic = time.time()
        rcnn_dict, vgg_dict, fei_caption_dict, msr_cap_dict, dummy = load_all_modules_dict(video_name)
        toc = time.time()
        print "Load all module time: ", toc-tic
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


        # sliding window   
        uniform_scores = []
        greedy_scores = []
        start_fids = []
        video_start_fid = 0
        video_end_fid = 0
        while True:

            if video_start_fid > video_len_f - 1:
                break  
            video_end_fid = min(video_start_fid + SERVER_STORAGE_FRAMES, video_len_f)
            #print 'start:', video_start_fid, ' end:', video_end_fid
 
            ''' Optimal '''
            tic = time.time()            
            optimal_frames = [os.path.join('/mnt/frames', video_name, str(x) + '.jpg') for x in range(video_start_fid, video_end_fid)] 
            optimal_tf_list = combine_all_modeldicts(vgg_dict, msr_cap_dict, rcnn_dict, fei_caption_dict, optimal_frames)
            optimal_tf = get_combined_tfs(optimal_tf_list)
            optimal_tf = remove_unimportantwords(optimal_tf)
            toc = time.time()
            print "Compute optimal for a window:", toc-tic
            #print optimal_tf
    
            ''' Uniform '''
            uni_range_fids = get_inrange_fids(video_start_fid, video_end_fid, uniform_frames)
            #print uni_range_fids
            uniform_inrange_frames = [os.path.join('/mnt/frames', video_name, str(x) + '.jpg') for x in uni_range_fids] 
            uniform_tf_list = combine_all_modeldicts(vgg_dict, msr_cap_dict, rcnn_dict, fei_caption_dict, uniform_inrange_frames)
            uniform_tf = get_combined_tfs(uniform_tf_list)
            uniform_score = detailed_measure(optimal_tf, uniform_tf) 
           

            ''' Greedy '''
            greedy_range_fids = get_inrange_fids(video_start_fid, video_end_fid, greedy_frames)
            #print greedy_range_fids
            greedy_inrange_frames = [os.path.join('/mnt/frames', video_name, str(x) + '.jpg') for x in greedy_range_fids] 
            greedy_tf_list = combine_all_modeldicts(vgg_dict, msr_cap_dict, rcnn_dict, fei_caption_dict, greedy_inrange_frames)
            #print greedy_tf_list
            greedy_tf = get_combined_tfs(greedy_tf_list)
            #greedy_tf = remove_unimportantwords(greedy_tf)
            '''
            for w in optimal_tf:
                if w not in greedy_tf:
                    print w, optimal_tf[w]
            '''
            greedy_score = detailed_measure(optimal_tf, greedy_tf) 
            #print greedy_tf 
            #print uniform_score, greedy_score

            uniform_scores += [uniform_score]
            greedy_scores += [greedy_score]
            start_fids += [video_start_fid]

            video_start_fid +=  SLIDE_SIZE_FRAMES 

     
        #plot_scores(video_name, start_fids, greedy_scores, uniform_scores, selected_frame_obj['picked_f'], [int(x.split('.')[0]) for x in uniform_frames], subsampled_rate, BASELINE_SAMPLERATE)
        with open(os.path.join(GREEDY_FOLDER, video_name + '_greedy_' + str(THRESH) + '_' + str(BASELINE_SAMPLERATE) + '.pickle'), 'wb') as fh:
            pickle.dump(greedy_scores, fh)
        with open(os.path.join(GREEDY_FOLDER, video_name + '_uniform_' + str(THRESH) + '_' + str(BASELINE_SAMPLERATE) + '.pickle'), 'wb') as fh:
            pickle.dump(uniform_scores, fh)
