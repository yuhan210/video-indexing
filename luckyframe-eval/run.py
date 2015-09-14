from nltk.corpus import stopwords
import numpy as np
import operator
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

'''
def uniform_subsample(video_name, DROP_FRAME_PERCENTAGE):
    
    if DROP_FRAME_PERCENTAGE % 5 != 0:
        print 'Drop frame percentage should be the multiple of 5 [Drop frame percentage =', DROP_FRAME_PERCENTAGE, ']'
        return

    dropped_frame = DROP_FRAME_PERCENTAGE/5
    # location of the dropped frames
    dropped_location = range(dropped_frame) 

    # frame name should be sorted 
    frame_names = os.listdir(os.path.join('/mnt/frames', video_name))
    frame_names = sorted(frame_names, key= lambda x: int(x.split('.')[0]))

    retained_frames = [] 
    for idx, frame_name in enumerate(frame_names):
    
        if idx % 20 in dropped_location:
            print 'drop', frame_name
            continue 
    
        print 'retain', frame_name
        retained_frames += [frame_name]

    return retained_frames
'''

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

    return combined_tfs

def detailed_measure(all_tf, subsampled_tf): 
    '''
    # of subsampled words/# of all words
    '''
    return len(subsampled_tf)/(len(all_tf) * 1.0)
    

def L1_dist(a_hist, b_hist): # incorrect

    assert(a_hist.shape == b_hist.shape)
    return np.linalg.norm((a_hist-b_hist), ord = 1)/(a_hist.shape[0] * .01)


def hist_measure(all_tf, subsampled_tf):
    '''
    similarity of the word distribution
    '''
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
     
    print 'Correlation (higher-> more similar) :', cv2.compareHist(all_hist_cv, sub_hist_cv, cv2.cv.CV_COMP_CORREL)
    print 'Chi-square (higher-> more similar) :', cv2.compareHist(all_hist_cv, sub_hist_cv, cv2.cv.CV_COMP_CHISQR)
    print 'L1 distance:', L1_dist(all_array, sub_array)
    ax = plt.subplot(2,1,1)
    plt.plot(range(len(all_hist)), all_array, 'o-r', label = 'Nonsubsampled') 
    plt.plot(range(len(sub_hist)), sub_array, 'x-b', label = 'Subsampled') 
    plt.plit(range(len(all_hist)), all_array - sub_array, 'o-k', label= 'L1 dist')
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

def combine_all_models(video_name, _vgg_data, _msr_data, _rcnn_data, _fei_data):

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
            for rcnn_idx, pred in enumerate(rcnn_data['pred']['text']):
                ## the confidence is higher than 10^(-3) and is not background
                if rcnn_data['pred']['conf'][rcnn_idx] > 0.0005 and pred.find('background') < 0:
                    rcnn_ws += [pred]
 
        vgg_ws = []
        if len(vgg_data) > 0:        
            vgg_ws = [w for w in vgg_data['pred']['text']]
    
        fei_ws = [] 
        if len(fei_data) > 0:
            fei_ws = removeStopWords(fei_data['candidate']['text'])
    
        msr_ws = [] 
        if len(msr_data) > 0:
            msr_ws = removeStopWordsFromWordlist(msr_data['words']['text'])

        words = {}
        for w in rcnn_ws:
            if w not in words:
                words[w] = 1
            else:
                words[w] += 1
        for w in vgg_ws:
            w = composeVGGWordnet(w)
            w = w.split('->')[-1] 
            if w not in words:
                words[w] = 1
            else:
                words[w] += 1
        
        for w in fei_ws:
            if w not in words:
                words[w] = 1
            else:
                words[w] += 1
    
        for w_idx, w in enumerate(msr_data['words']['text']):
            if w not in STOPWORDS and w.find('background') < 0:
                if w not in words:
                    words[w] = 1
                else:
                    words[w] += 1

        tf_list += [{'frame_name': frame_name, 'tf': words}]

    return tf_list

   

def subsample_tf_list(selected_frames, all_tf_list):
   
    tfs = []
    for fid, frame_name in enumerate(selected_frames):
        #print fid, '/', len(selected_frames)
        frame_id = int(frame_name.split('.')[0])
        
        tf = filter(lambda x: int(x['frame_name'].split('.')[0]) == frame_id, all_tf_list)
        tfs += [tf[0]]  

    return tfs

#def hist_timely_measure():


#def top_word_timely_measure():



if __name__ == "__main__":
   
    VIDEO_LIST = '/mnt/test_video.txt'
    videos = open(VIDEO_LIST).read().split()

    for video_name in videos:
        
        if not os.path.exists(os.path.join('/mnt/tags/rcnn-info-all', video_name + '_rcnnrecog.json')) or not os.path.exists(os.path.join('/mnt/tags/vgg-classify-all', video_name + '_recog.json')) or not os.path.exists(os.path.join('/mnt/tags/msr-caption-all', video_name + '_msrcap.json')) or not os.path.exists(os.path.join('/mnt/tags/fei-caption-all', video_name + '_5_caption.json')):
            continue
      
        print video_name
        
        # load tags from all DNN modules 
        _vgg_data = load_video_recog('/mnt/tags/vgg-classify-all', video_name)
        _fei_caption_data = load_video_caption('/mnt/tags/fei-caption-all', video_name)
        _msr_cap_data = load_video_msr_caption('/mnt/tags/msr-caption-all', video_name)
        _rcnn_data = load_video_rcnn('/mnt/tags/rcnn-info-all', video_name)
      
 
        # compose video term freq (a list of dicts)
        frame_names = os.listdir(os.path.join('/mnt/frames', video_name))
        frame_names = sorted(frame_names, key= lambda x: int(x.split('.')[0]))
        all_tfs_list = combine_all_models(video_name, _vgg_data, _msr_cap_data, _rcnn_data, _fei_caption_data)
        all_tf = get_combined_tfs(tf_list)
 
        # uniformly subsample frames 
        FRAME_RETAIN_RATE = 10/100. 
        #for frame_rate in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        for retained_frame_rate in [0.01, 0.05, 0.1, 0.2, 0.5]:
            retained_frames = naive_subsample_frames(frame_names, retained_frame_rate)
            subsampled_tfs_list = subsample_tf_list(retained_frames, all_tfs_list) 
            subsampled_tf = get_combined_tfs(subsampled_tfs_list)

            print retained_frame_rate, detailed_measure(all_tf, subsampled_tf)
            hist_measure(all_tf, subsampled_tf)
   
           # hist_timely_measure()
           # top_word_timely_measure()
        # frame diff (scene changes)
        
        break 
