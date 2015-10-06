from nltk.corpus import stopwords
import scipy.spatial.distance
import numpy as np
import operator
import pickle
import shutil
import math
import time
import sys
TOOL_PATH = '/home/t-yuche/clustering/tools'
sys.path.append(TOOL_PATH)
from utils import *
import matplotlib
import matplotlib.pyplot as plt
try:
    plt.style.use('ggplot')
except:
    pass
import cv2

STOPWORDS = ['inside', 'near', 'two', 'day', 'front', u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not', u'only', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don', u'should', u'now']


def removeStopWordsFromWordlist(list_ws):

    ws = []
    for s in list_ws:
        if s not in stopwords.words('english') and s.find('background') < 0 and s not in STOPWORDS:
            ws += [s]

    return ws


def removeStopWords(list_str):
    ws = []
    for s in list_str:
       ws.extend( [w for w in s.split(' ') if w not in stopwords.words('english') and w.find('background') < 0  and w not in STOPWORDS] )

    return ws

def getwnid(w):
    with open('synset_words.txt') as f:
        for l in f.readlines():
            wnid = l.strip().split(' ')[0]
            name = [x.strip() for x in ' '.join(l.strip().split(' ')[1:]).split(',')][0]
            
            if name == w:
                return wnid

#def uniform_subsample(video_name, DROP_FRAME_PERCENTAGE):
#    
#    if DROP_FRAME_PERCENTAGE % 5 != 0:
#        print 'Drop frame percentage should be the multiple of 5 [Drop frame percentage =', DROP_FRAME_PERCENTAGE, ']'
#        return
#
#    dropped_frame = DROP_FRAME_PERCENTAGE/5
#    # location of the dropped frames
#    dropped_location = range(dropped_frame) 
#
#    # frame name should be sorted 
#    frame_names = os.listdir(os.path.join('/mnt/frames', video_name))
#    frame_names = sorted(frame_names, key= lambda x: int(x.split('.')[0]))
#
#    retained_frames = [] 
#    for idx, frame_name in enumerate(frame_names):
#    
#        if idx % 20 in dropped_location:
#            print 'drop', frame_name
#            continue 
#    
#        print 'retain', frame_name
#        retained_frames += [frame_name]
#
#    return retained_frames


def naive_subsample_frames(all_frames, FRAME_RETAIN_RATE):

    n_picked_frames = len(all_frames) * FRAME_RETAIN_RATE
    step = n_picked_frames/((len(all_frames) ) * 1.0)
    track = 0
    counter = 0
    retained_frames = [] 
    for idx, frame_name in enumerate(all_frames):
        if int(track) == counter:
            retained_frames += [frame_name]
            counter += 1
        track += step

    return retained_frames

def get_combined_tfs(tfs_dict):

    combined_tfs = {}
    for d in tfs_dict:
        for w in d['tf']:
            if w not in combined_tfs:
                combined_tfs[w] = 1
            else:
                combined_tfs[w] += 1

    #combined_tfs = keep_nouns(combined_tfs)
    return combined_tfs

'''
# of subsampled words/# of all words
'''
def detailed_measure(all_tf, subsampled_tf): 
    return len(subsampled_tf)/(len(all_tf) * 1.0)
    

def L1_dist(a_hist, b_hist): # incorrect

    assert(a_hist.shape == b_hist.shape)
    return np.linalg.norm((a_hist-b_hist), ord = 1)/(a_hist.shape[0] * .01)


def cos_similarity(a_hist, b_hist):
    return 1 - scipy.spatial.distance.cosine(a_hist, b_hist) 
'''
similarity of the word distribution
'''
def hist_measure(all_tf, subsampled_tf, plot_fig = False):
    sorted_all_tf = sorted(all_tf.items(), key=operator.itemgetter(1)) # becomes tuple
 
    # create subsampled histogram
    sub_hist = []
    all_hist = []
    for item in sorted_all_tf:
        all_hist += [item[1]]
        if item[0] in subsampled_tf:
            sub_hist += [subsampled_tf[item[0]]]
        else:
            sub_hist += [0]

    # correlation
    all_array = np.array(all_hist)
    sub_array = np.array(sub_hist)
    all_array = all_array / (np.sum(all_array) * 1.0)
    sub_array = sub_array / (np.sum(sub_array) * 1.0)
  
    all_hist_cv = all_array.ravel().astype('float32') 
    sub_hist_cv = sub_array.ravel().astype('float32') 
     
    correlation_score =  cv2.compareHist(all_hist_cv, sub_hist_cv, cv2.cv.CV_COMP_CORREL)
    chisq_dist = cv2.compareHist(all_hist_cv, sub_hist_cv, cv2.cv.CV_COMP_CHISQR)
    l1_dist = L1_dist(all_array, sub_array)
    cos_sim = cos_similarity(all_array, sub_array)
    '''
    print 'Correlation (higher-> similar) :', cv2.compareHist(all_hist_cv, sub_hist_cv, cv2.cv.CV_COMP_CORREL)
    print 'Chi-square (smaller-> similar) :', cv2.compareHist(all_hist_cv, sub_hist_cv, cv2.cv.CV_COMP_CHISQR)
    print 'L1 distance (smaller -> similar):', L1_dist(all_array, sub_array)
    '''
    if plot_fig:
        ax = plt.subplot(2,1,1)
        plt.plot(range(len(all_hist)), all_array, 'o-r', label = 'Nonsubsampled') 
        plt.plot(range(len(sub_hist)), sub_array, 'x-b', label = 'Subsampled') 
        plt.plot(range(len(all_hist)), abs(all_array - sub_array), 'o-k', label= 'L1 dist')
        plt.legend()
        plt.xlabel('Word Index')
        plt.ylabel('Word Count (#)')
        ax.set_title('Histogram')
        
        ax = plt.subplot(2,1,2)
        plt.plot(sub_hist, all_hist, 'o')
        plt.xlabel('Subsampled Word Count (#)')
        plt.ylabel('Nonsubsampled Word Count (#)')
        ax.set_title('Correlation')
        plt.show()

    return l1_dist, correlation_score, chisq_dist, cos_sim

    
def composeVGGWordnet(w, wordnet_tree_path = 'new_synset_word.txt'):
    
    wnid = getwnid(w)
    # read tree
    word_tree = json.load(open('synset_word_tree')) 
    stop_data = json.load(open(wordnet_tree_path))

    for key in stop_data:
        match_key = key.split('_ ')[0]
        if match_key == w:
            till = stop_data[key]
   
    if type(till) is list:
        till = till[0]
    
    # process till string
    till = [x.strip() for x in till.split(',')][0]

    output_w = [w]
    if till == w:
     #   print 'direct match word:', w, 'till:', till, 'output_w:', output_w
        return '->'.join(output_w)

    # compose string 
    for idx, e in enumerate(word_tree[wnid][0]):
        
        match_e =  [x.strip() for x in e[-1].split(',')][0]
     #   print '\t\t', 'till:', till, 'cur_match:', match_e
        output_w += [e[-1].split(',')[0]]
        if match_e == till:
            break
    #print 'word:', w, 'till:', till, 'output_w:', output_w
    return '->'.join(output_w)
     

#def weighted_combine_models(video_name, selected_frames, _vgg_data, _msr_data, _rcnn_data, _fei_data):

def subsample_tf_list(selected_frames, all_tf_list):
   
    tfs = []
    for fid, frame_name in enumerate(selected_frames):
        #print fid, '/', len(selected_frames)
        frame_id = int(frame_name.split('.')[0])
        
        tf = filter(lambda x: int(x['frame_name'].split('.')[0]) == frame_id, all_tf_list)
        tfs += [tf[0]]  

    return tfs

def hist_timely_measure(all_tfs_list, subsampled_tfs_list, t = 5):
    '''
    Check hist distance every t secs 
    '''

    chunk_size = t * 30
    
    # check if there's at least one chunk
    if len(all_tfs_list) < chunk_size or len(subsampled_tfs_list) < chunk_size:
        return -1, -1 

    
    chunk_num = 1
    nonsubsampled_idx = 0 
    subsampled_idx = 0 
    
    nonsub_chunk = {}
    sub_chunk = {}
    timely_dist = {}
    timely_corr = {}

    while True:

        #print 'nonsub id:', nonsubsampled_idx, ' subsampled_id:', subsampled_idx
        if nonsubsampled_idx == len(all_tfs_list) and subsampled_idx == len(subsampled_tfs_list):
            # Ending criteria
            break
    
        nonsub_cur_fid = int(all_tfs_list[nonsubsampled_idx]['frame_name'].split('.')[0])

        update_subchunk = True
        if subsampled_idx < len(subsampled_tfs_list):
            sub_cur_fid = int(subsampled_tfs_list[subsampled_idx]['frame_name'].split('.')[0])
        else:
            update_subchunk = False 

        #print 'chunk region:', chunk_num * chunk_size
        #print 'nonsub fid:', nonsub_cur_fid, ' subsampled_id:', sub_cur_fid

        if nonsub_cur_fid < chunk_num * chunk_size and nonsub_cur_fid >= (chunk_num - 1) * chunk_size:
            #print 'update nonsubsampled idx'
            for w in all_tfs_list[nonsubsampled_idx]['tf']:
                if w not in nonsub_chunk:
                    nonsub_chunk[w] = 1
                else:
                    nonsub_chunk[w] += 1
            nonsubsampled_idx += 1 
        
             
        if update_subchunk and sub_cur_fid < chunk_num * chunk_size and sub_cur_fid >= (chunk_num - 1) * chunk_size:
            #print 'update subsampled idx'
            for w in subsampled_tfs_list[subsampled_idx]['tf']:
                if w not in sub_chunk:
                    sub_chunk[w] = 1
                else:
                    sub_chunk[w] += 1
            subsampled_idx += 1
            
        # process this chunk and advance to the next chunk 
        # if no more fids or next fid both >= chunk_num * chunk_size
        if (nonsubsampled_idx == len(all_tfs_list) and subsampled_idx == len(subsampled_tfs_list) ) or \
           (subsampled_idx != len(subsampled_tfs_list) and nonsubsampled_idx != len(all_tfs_list) and int(subsampled_tfs_list[subsampled_idx]['frame_name'].split('.')[0]) >= chunk_num * chunk_size and int (all_tfs_list[nonsubsampled_idx]['frame_name'].split('.')[0]) >= chunk_num * chunk_size) or \
           (subsampled_idx == len(subsampled_tfs_list) and int(all_tfs_list[nonsubsampled_idx]['frame_name'].split('.')[0]) >= chunk_num * chunk_size): 
 
       #if nonsubsampled_idx >= chunk_num * chunk_size and subsampled_idx >= chunk_num * chunk_size:
            # compute nonsub_chunk and sub_chunk dist
            if len(sub_chunk) == 0:
                #print 'Using previous chunk'
                sub_chunk = prev_sub_chunk

            #print sub_chunk, nonsub_chunk

            sorted_nonsub_tf = sorted(nonsub_chunk.items(), key=operator.itemgetter(1)) 
            nonsub_hist = []
            sub_hist = []
            for item in sorted_nonsub_tf:
                nonsub_hist += [item[1]]
                if item[0] in sub_chunk:
                    sub_hist += [sub_chunk[item[0]]]
                else:
                    sub_hist += [0]

            nonsub_array = np.array(nonsub_hist)
            sub_array = np.array(sub_hist)
            nonsub_array = nonsub_array / (np.sum(nonsub_array) * 1.0)
            sub_array = sub_array / (np.sum(sub_array) * 1.0)
            nonsub_hist_cv = nonsub_array.ravel().astype('float32') 
            sub_hist_cv = sub_array.ravel().astype('float32') 
             
            corr_score =  cv2.compareHist(nonsub_hist_cv, sub_hist_cv, cv2.cv.CV_COMP_CORREL)
            dist = L1_dist(nonsub_array, sub_array)
            timely_dist[chunk_num] = dist
            timely_corr[chunk_num] = corr_score
            
            prev_sub_chunk = sub_chunk
            chunk_num += 1
            nonsub_chunk = {}
            sub_chunk = {}
           
    ave_dist = sum(map(lambda x:timely_dist[x], timely_dist))/(len(timely_dist) * 1.0)
    ave_corr = sum(map(lambda x:timely_corr[x], timely_corr))/(len(timely_corr) * 1.0)
    return ave_dist, ave_corr

def top_word_timely_measure(k, all_tf, all_tfs_list, subsampled_tfs_list):

    sorted_all_tf = sorted(all_tf.items(), key=operator.itemgetter(1)) # becomes tuple
    top_k_w = [x[0] for x in sorted_all_tf[-(k+1):-1]]

    # TODO:re-occurence of an object
    # diff by frame
    #print top_k_w
    nonsubsampled_occur_time = {}
    subsampled_occur_time = {}
    
    for keyword in top_k_w:
        for d in all_tfs_list:
            tf = d['tf']
            frame_name = d['frame_name']
            if keyword in tf:
                nonsubsampled_occur_time[keyword] = int(frame_name.split('.')[0])
                break

    for keyword in top_k_w:
        keyword_found = False
        for idx, d in enumerate(subsampled_tfs_list):
            tf = d['tf']  
            frame_name = d['frame_name']
            if keyword in tf:
                keyword_found = True
                subsampled_occur_time[keyword] = int(frame_name.split('.')[0])
                break

        # TODO: Make suer this makes sense -- if subsampled never gets this keyword, its occurence is the last frame
        if not keyword_found:
            # its occurence is at the end of the video
            subsampled_occur_time[keyword] = int(all_tfs_list[-1]['frame_name'].split('.')[0])


    occur_dist = {} # in ms

    # compute distance
    ave_dist = 0.0
    for keyword in top_k_w:
        dist = (subsampled_occur_time[keyword] - nonsubsampled_occur_time[keyword]) * 33.33
        occur_dist[keyword] = dist
        ave_dist += dist

    ave_dist /= len(top_k_w)
    
    return ave_dist, occur_dist 


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

def store_keyframes(video_name, keyframe_ids, outfolder = './keyframes'):
    
    frames = [os.path.join('/mnt/frames', video_name, x) for x in sorted(os.listdir(os.path.join('/mnt/frames', video_name)), key= lambda x: int(x.split('.')[0]))]

    prev_frame_jump = 3
    if os.path.exists(outfolder):
        shutil.rmtree(outfolder)    
    os.makedirs(outfolder)

    for fid, frame_path in enumerate(frames):
        if fid == 0:
            continue

        if fid in keyframe_ids:# keyframe
            idx = keyframe_ids.index(fid)

            frame_name = frame_path.split('/')[-1] 
            dst_filename = str(idx) + '_k_' + frame_name
            shutil.copyfile(frame_path, os.path.join(outfolder, dst_filename))
            
            prev_framename = frames[fid-prev_frame_jump].split('/')[-1] 
            prev_dst_filename = str(idx) + '_kp_' + prev_framename
            shutil.copyfile(frames[fid-prev_frame_jump], os.path.join(outfolder, prev_dst_filename))


if __name__ == "__main__":
   
    VIDEO_LIST = '/mnt/video_list.txt'
    videos = open(VIDEO_LIST).read().split()

    fh = open('./sample_log.txt', 'w') 
    for video_name in videos:
       
 
        if not os.path.exists(os.path.join('/mnt/tags/rcnn-info-all', video_name + '_rcnnrecog.json')) or not os.path.exists(os.path.join('/mnt/tags/vgg-classify-all', video_name + '_recog.json')) or not os.path.exists(os.path.join('/mnt/tags/msr-caption-all', video_name + '_msrcap.json')) or not os.path.exists(os.path.join('/mnt/tags/fei-caption-all', video_name + '_5_caption.json')):
            continue
      
        print video_name

        # load tags from all DNN modules 
        _vgg_data = load_video_recog('/mnt/tags/vgg-classify-all', video_name)
        _fei_caption_data = load_video_caption('/mnt/tags/fei-caption-all', video_name)
        _msr_cap_data = load_video_msr_caption('/mnt/tags/msr-caption-all', video_name)
        _rcnn_data = load_video_rcnn('/mnt/tags/rcnn-info-all', video_name)
     
        _turker_data = load_video_processed_turker(video_name)
 
        # compose video term freq (a list of dicts)
        frame_names = os.listdir(os.path.join('/mnt/frames', video_name))
        frame_names = sorted(frame_names, key= lambda x: int(x.split('.')[0]))
        all_tfs_list = combine_all_models(video_name, _vgg_data, _msr_cap_data, _rcnn_data, _fei_caption_data)
        all_tf = get_combined_tfs(all_tfs_list)
   
        hysteresis_num = 2
        # All detailed measure 
        DETAIL_THRESHOLD = 0.4
        start_idx = 0
        end_idx = 0
        cur_tfs_list = [all_tfs_list[0]]
        detailed_values = []
        detailed_picked_fid = [0]
        prev_bad_frame = False
 
        while True:
            if end_idx >= len(all_tfs_list):
                break
       
            gt_tfs_list = all_tfs_list[start_idx : end_idx + 1]
            gt_tf = get_combined_tfs(gt_tfs_list)

            cur_tf = get_combined_tfs(cur_tfs_list)

            detailed_m = detailed_measure(gt_tf, cur_tf)
              
            detailed_values += [detailed_m]

            if detailed_m < DETAIL_THRESHOLD: #and (end_idx + hysteresis_num) < len(all_tfs_list): # taking min(+hysteresis_num, good frames)

                # eat some hysteresis frames
                for i in xrange(hysteresis_num):
                     
                    end_idx += 1
                    if end_idx >= len(all_tfs_list):
                        break

                    gt_tfs_list = all_tfs_list[start_idx : end_idx + 1]
                    gt_tf = get_combined_tfs(gt_tfs_list)

                    if i == hysteresis_num - 1:

                        if prev_bad_frame == False:
                             
                            detailed_picked_fid += [end_idx] 
                            cur_tfs_list = [all_tfs_list[end_idx]]
                            cur_tf = get_combined_tfs(cur_tfs_list)
                            start_idx = end_idx

                            if isbadframe(_turker_data, all_tfs_list[end_idx]):
                                prev_bad_frame = True
                        else: 
                            # eat bad frames
                            while True:

                                if end_idx >= len(all_tfs_list):
                                    break

                                if isbadframe(_turker_data, all_tfs_list[end_idx]):
                                
                                    gt_tfs_list = all_tfs_list[start_idx : end_idx + 1]
                                    gt_tf = get_combined_tfs(gt_tfs_list)
                                    detailed_m = detailed_measure(gt_tf, cur_tf)
                                    detailed_values += [detailed_m]
                                    end_idx += 1
                                else:
                                    # pick the frame
                                    detailed_picked_fid += [end_idx] 
                                    cur_tfs_list = [all_tfs_list[end_idx]]
                                    cur_tf = get_combined_tfs(cur_tfs_list)
                                    start_idx = end_idx
                                    prev_bad_frame = False
                                    break
                        
                        gt_tfs_list = all_tfs_list[start_idx : end_idx + 1]
                        gt_tf = get_combined_tfs(gt_tfs_list)
                    
                    detailed_m = detailed_measure(gt_tf, cur_tf)
                    detailed_values += [detailed_m]

            end_idx += 1 
        print len(detailed_picked_fid) 

        with open(os.path.join('greedy-log', video_name + '_gtframe.pickle'), 'wb') as gt_fh:
            gt_fh({'picked_f': detailed_picked_fid, 'total_frame': len(all_tfs_list), 'picked_rate': len(detailed_picked_fid)/(len(all_tfs_list) * 1.0), gt_fh)
   
        # All histogram correlation       
        '''
        CORRE_THRESHOLD = 0.87
        start_idx = 0
        end_idx = 0
        cur_tfs_list = [all_tfs_list[0]]
        hist_measures = []
        hist_measure_picked_fid = [0]
        while True:
            if end_idx == len(all_tfs_list):
                break
       
            gt_tfs_list = all_tfs_list[start_idx : end_idx + 1]
            gt_tf = get_combined_tfs(gt_tfs_list)

            cur_tf = get_combined_tfs(cur_tfs_list)
            l1_dist, corr_score, chisq_dist, cos_sim = hist_measure(gt_tf, cur_tf)
             
            hist_measures += [cos_sim]

            if cos_sim < CORRE_THRESHOLD and (end_idx + hysteresis_num) < len(all_tfs_list): 
                
                for i in xrange(hysteresis_num):

                    end_idx += 1
                    gt_tfs_list = all_tfs_list[start_idx : end_idx + 1]
                    gt_tf = get_combined_tfs(gt_tfs_list)
               
                    if i == hysteresis_num - 1:
                        hist_measure_picked_fid += [end_idx]
                
                        cur_tfs_list = [all_tfs_list[end_idx]]
                        cur_tf = get_combined_tfs(cur_tfs_list)

                        start_idx = end_idx
                        gt_tfs_list = all_tfs_list[start_idx : end_idx + 1]
                        gt_tf = get_combined_tfs(gt_tfs_list)

                    l1_dist, corr_score, chisq_dist, cos_sim = hist_measure(gt_tf, cur_tf)
                    hist_measures += [cos_sim]
                    
            end_idx += 1 
        print len(hist_measure_picked_fid) 
        '''
        '''
        # Timeliness correlation
        T_CORRE_THRESHOLD = 0.9
        start_idx = 0
        end_idx = 0
        cur_tfs_list = [all_tfs_list[0]]
        t_hist_measures = []
        t_hist_measure_picked_fid = [0]
        while True:
            if end_idx == len(all_tfs_list):
                break
                  
            gt_tfs_list = all_tfs_list[start_idx : end_idx + 1]
            
            ave_dist, ave_corr = hist_timely_measure(gt_tfs_list, cur_tfs_list)
             

            if end_idx > 0 and ave_corr < T_CORRE_THRESHOLD:
                #cur_tfs_list += [all_tfs_list[end_idx]]
                start_idx = end_idx

                gt_tfs_list = all_tfs_list[start_idx : end_idx + 1]
                
                cur_tfs_list = [all_tfs_list[end_idx]]
        
                ave_dist, ave_corr = hist_timely_measure(gt_tfs_list, cur_tfs_list)
                t_hist_measure_picked_fid += [end_idx] 
 
            t_hist_measures += [ave_corr]
            end_idx += 1 
        print len(t_hist_measure_picked_fid) 
        
        # top k 
        TOPK_THRESHOLD = 1000
        start_idx = 0
        end_idx = 0
        cur_tfs_list = [all_tfs_list[0]]
        topk_dists = []
        top_k_picked_fid = [0]
        while True:
            if end_idx == len(all_tfs_list):
                break
       
            gt_tfs_list = all_tfs_list[start_idx : end_idx + 1]

            gt_tf = get_combined_tfs(gt_tfs_list)
            ave_dist, dummy= top_word_timely_measure(10, gt_tf, gt_tfs_list, cur_tfs_list)
             

            if end_idx > 0 and ave_dist > TOPK_THRESHOLD:
                #cur_tfs_list += [all_tfs_list[end_idx]]
                start_idx = end_idx

                gt_tfs_list = all_tfs_list[start_idx : end_idx + 1]
                gt_tf = get_combined_tfs(gt_tfs_list)

                cur_tfs_list = [all_tfs_list[end_idx]]
                
                ave_dist, dummy= top_word_timely_measure(10, gt_tf, gt_tfs_list, cur_tfs_list)
                top_k_picked_fid += [end_idx]
            
            topk_dists += [ave_dist]
            end_idx += 1 
        print len(top_k_picked_fid) 
        '''

        #store_keyframes(video_name, detailed_picked_fid, os.path.join('./keyframes', video_name, 'detailed'))
        #store_keyframes(video_name, hist_measure_picked_fid, os.path.join('./keyframes', video_name, 'hist_measure'))

        fh.write(video_name + '\n')
        fh.write('Detailed:' + str(len(detailed_picked_fid)/(len(detailed_values) * 1.0)) + '\n')
        #fh.write('Hist:' + str(len(hist_measure_picked_fid)/(len(detailed_values) * 1.0)) + '\n\n')
        
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.set_title('# subsampled distinct word/# nonsubsampled distinct word (bigger -> more similar)')
        
        plt.text(len(detailed_values)/2, DETAIL_THRESHOLD - 0.2, 'Portion of selected frames:' + str(len(detailed_picked_fid)/(len(detailed_values) * 1.0)),fontsize = 20)
        plt.plot(range(len(detailed_values)), detailed_values)
        plt.scatter(detailed_picked_fid, [1 for x in detailed_picked_fid], marker = (5,1))
        plt.plot([0, len(detailed_values)], [DETAIL_THRESHOLD, DETAIL_THRESHOLD], 'k-')
        ax.set_xlabel('Frame ID')
        ax.set_ylabel('# subsampled distinct word/# nonsubsampled distinct word')
        ax.set_ylim([0,1]) 
        plt.show()
        '''
        fig = plt.figure(2)
        ax = fig.add_subplot(111) 
        ax.set_title('Histogram Correlation (bigger -> more similar)')

        plt.text(len(detailed_values)/2, CORRE_THRESHOLD - 0.2, 'Portion of selected frames:' + str(len(hist_measure_picked_fid)/(len(detailed_values) * 1.0)),fontsize = 20)
        plt.plot(range(len(hist_measures)), hist_measures) 
        plt.scatter(hist_measure_picked_fid, [1 for x in hist_measure_picked_fid], marker = (5,1))
        plt.plot([0, len(detailed_values)], [CORRE_THRESHOLD, CORRE_THRESHOLD], 'k-')
        ax.set_xlabel('Frame ID')
        ax.set_ylabel('Histogram Correlation')
        ax.set_ylim([0,1]) 
        ''' 
        ''' 
        plt.figure(3) 
        plt.title('5-sec Histogram Correlation (bigger -> more similar)')
        plt.plot(range(len(t_hist_measures)), t_hist_measures) 
        plt.scatter(t_hist_measure_picked_fid, [1 for x in t_hist_measure_picked_fid], marker = (5,1))
        plt.xlabel('Frame ID')
        plt.ylabel('5-sec Histogram Correlation')
        
        plt.figure(4) 
        plt.title('Top k word occurence time difference (smaller->more similar)')
        plt.plot(range(len(topk_dists)), topk_dists) 
        plt.scatter(top_k_picked_fid, [1 for x in top_k_picked_fid], marker = (5,1))
        plt.xlabel('Frame ID')
        plt.ylabel('Ave occurence time diff (ms)')
        plt.show()
        '''
        
        # frame difference
        #framediff_retained_frames = framediff_select(video_name, frame_names) 
        #framediff_tfs_list = subsample_tf_list(framediff_retained_frames, all_tfs_list) 
        #framediff_tf = get_combined_tfs(framediff_tfs_list)
        #print '-----------------FRAMEDIFF-------------------------'
        #print '<retained rate>', len(framediff_retained_frames)/(len(frame_names) * 1.0) 
        #print '1. # subsampled distinct word/nonsubsampled>', detailed_measure(all_tf, framediff_tf)
        #l1_dist, correl_score, chisq_dist = hist_measure(all_tf, framediff_tf)
        #print '2. hist measure'
        #print '\t L1-dist (smaller -> similar)', l1_dist
        #print '\t Correlation (smaller -> similar)', correl_score
        #print '\t Chi-sq (bigger -> similar)', chisq_dist

        #k = 10
        #ave_dist, occur_dist = top_word_timely_measure(k, all_tf, all_tfs_list, framediff_tfs_list)
        #print '3. top', k, 'word timely measure (ms):', ave_dist
        #ave_dist, timely_dist = hist_timely_measure(all_tfs_list, framediff_tfs_list)
        #print '4. timely hist measure:', ave_dist
         
 
        # uniformly subsample frames 
        #for frame_rate in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        #for retained_frame_rate in [0.01, 0.03, 0.05, 0.1, 0.2, 0.5]:
        #    retained_frames = naive_subsample_frames(frame_names, retained_frame_rate)
        #    subsampled_tfs_list = subsample_tf_list(retained_frames, all_tfs_list) 
        #    subsampled_tf = get_combined_tfs(subsampled_tfs_list)
        #    print '-----------------UNIFORM-------------------------'
        #    print '<retained rate>', retained_frame_rate
        #    print '1. # subsampled distinct word/nonsubsampled>', detailed_measure(all_tf, subsampled_tf)
        #    l1_dist, correl_score, chisq_dist = hist_measure(all_tf, subsampled_tf)
        #    print '2. hist measure'
        #    print '\t L1-dist (smaller -> similar)', l1_dist
        #    print '\t Correlation (smaller -> similar)', correl_score
        #    print '\t Chi-sq (bigger -> similar)', chisq_dist

        #    ave_dist,  = top_word_timely_measure(k, all_tf, all_tfs_list, subsampled_tfs_list)
        #    print '3. top', k, 'word timely measure (ms):', ave_dist
        #    ave_dist, ave_corr = hist_timely_measure(all_tfs_list, subsampled_tfs_list)
        #    print '4. timely hist measure:', ave_dist
        #
                
               
    fh.close() 
