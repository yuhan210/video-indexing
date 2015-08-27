from nltk.corpus import stopwords
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

STOPWORDS = ['inside', 'near', 'two', 'day', 'front']

def removeStopWordsFromWordlist(list_ws):

    ws = []
    for s in list_ws:
        if s not in stopwords.words('english') and s.find('background') < 0 and s not in STOPWORDS:
            ws += [s]

    return ws


def getwnid(w):

    with open('synset_words.txt') as f:
        for l in f.readlines():
            wnid = l.strip().split(' ')[0]
            name = [x.strip() for x in ' '.join(l.strip().split(' ')[1:]).split(',')][0]
            
            if name == w:
                return wnid

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


def naive_subsample_frames(video_name, FRAME_RETAIN_RATE):

    frame_names = os.listdir(os.path.join('/mnt/frames', video_name))
    frame_names = sorted(frame_names, key= lambda x: int(x.split('.')[0]))

    n_picked_frames = len(frame_names) * FRAME_RETAIN_RATE
    step = n_picked_frames/((len(frame_names) ) * 1.0)
    track = 0
    counter = 0
    retained_frames = [] 
    for idx, frame_name in enumerate(frame_names):
        if int(track) == counter:
            retained_frames += [frame_name]
            counter += 1
        track += step

    return retained_frames



#def get_all_words():

def detailed_measure(all_tf, subsampled_tf): 
    '''
    # of subsampled words/# of all words
    '''

    return len(subsampled_tf)/(len(all_tf) * 1.0)
    


#def hist_measure():
'''
similarity of the word distribution
'''

#def time_dist_measure():
'''
'''

#def combine_cnns():
    
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
     

def getSuggestedChoices(rcnn_data, vgg_data, fei_caption_data, msr_caption_data, frame_name):

    start_idx = int(frame_name.split('.')[0])

    rcnn_data = filter(lambda x: int(x['image_path'].split('/')[-1].split('.')[0]) == start_idx, rcnn_data) 
    vgg_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == start_idx, vgg_data) 
    fei_caption_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == start_idx, fei_caption_data) 
    msr_caption_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == start_idx, msr_caption_data) 
 
    labels = []
    idx = 0
    
    ## process fast-rcnn
    rcnn_ws = []
    if len(rcnn_data) > 0:
        for rcnn_idx, pred in enumerate(rcnn_data[idx]['pred']['text']):

            ## the confidence is higher than 10^(-3) and is not background
            if rcnn_data[idx]['pred']['conf'][rcnn_idx] > 0.0005 and  pred.find('background') < 0:
                rcnn_ws += [pred]

    vgg_ws = []
    if len(vgg_data) > 0:        
        ## process vgg
        vgg_ws = [w for w in vgg_data[idx]['pred']['text']]
   
 
    fei_caption_ws = [] 
    if len(fei_caption_data) > 0:
        ## process neuraltalk
        fei_caption_ws = removeStopWords(fei_caption_data[idx]['candidate']['text'])


    msr_caption_ws = [] 
    if len(msr_caption_data) > 0:
        ## process msr captioning
        msr_caption_ws = removeStopWordsFromWordlist(msr_caption_data[idx]['words']['text'])


    words = {}
    for w in rcnn_ws:
        if w not in words:
            words[w] = 4
        else:
            words[w] += 4

    for w in vgg_ws:
    # compose vgg word
        w = composeVGGWordnet(w)
        if w not in words:
            words[w] = 4
        else:
            words[w] += 4

    for w in fei_caption_ws:
        if w not in words:
            words[w] = 1
        else:
            words[w] += 1
    
    for w_idx, w in enumerate(msr_caption_data[idx]['words']['text']):
        score = float(msr_caption_data[idx]['words']['prob'][w_idx])
        if w not in stopwords.words('english') and w.find('background') < 0:
            if w not in words:
                words[w] = max(1, 5 + score)
            else:
                words[w] += max(1, 5 + score)


    words = sorted(words.items(), key=lambda x: x[1], reverse=True)    
    labels = [x for x in words]
    return labels 
    



def naive_combine_models(video_name, selected_frames, _vgg_data, _msr_data, _rcnn_data):
    
    words = {}
    for fid, frame_name in enumerate(selected_frames):
        print fid, '/', len(selected_frames)
        frame_id = int(frame_name.split('.')[0])
        
        rcnn_data = filter(lambda x: int(x['image_path'].split('/')[-1].split('.')[0]) == frame_id, _rcnn_data)
        vgg_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == frame_id, _vgg_data)
        msr_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == frame_id, _msr_data)
        
        # combine words
        rcnn_ws = []
        if len(rcnn_data) > 0:
            for rcnn_idx, pred in enumerate(rcnn_data[0]['pred']['text']):

                ## the confidence is higher than 10^(-3) and is not background
                if rcnn_data[0]['pred']['conf'][rcnn_idx] > 0.0005 and  pred.find('background') < 0:
                    rcnn_ws += [pred]
         
 
        vgg_ws = []
        if len(vgg_data) > 0:        
            ## process vgg
            vgg_ws = [w for w in vgg_data[0]['pred']['text']]


        msr_caption_ws = [] 
        if len(msr_data) > 0:
            ## process msr captioning
            msr_caption_ws = removeStopWordsFromWordlist(msr_data[0]['words']['text'])

        
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
        
        for w_idx, w in enumerate(msr_data[0]['words']['text']):
            score = float(msr_data[0]['words']['prob'][w_idx])
            if w not in stopwords.words('english') and w.find('background') < 0:
                if w not in words:
                    words[w] = 1
                else:
                    words[w] += 1

    return words

if __name__ == "__main__":
   
    VIDEO_LIST = '/mnt/video_list.txt'
    videos = open(VIDEO_LIST).read().split()

    for video_name in videos:
        print video_name
        
        if not os.path.exists(os.path.join('/mnt/tags/rcnn-info-all', video_name + '_rcnnrecog.json')) or not os.path.exists(os.path.join('/mnt/tags/vgg-classify-all', video_name + '_recog.json')) or not os.path.exists(os.path.join('/mnt/tags/msr-caption-all', video_name + '_msrcap.json')):
            continue
      
        #or not os.path.exists(os.path.join('/mnt/tags/fei-caption-all', video_name)) 

        # load tags 
        _vgg_data = load_video_recog('/mnt/tags/vgg-classify-all', video_name)
        #_fei_caption_data = load_video_caption('/mnt/tags/fei-caption-all', video_name)
        _msr_cap_data = load_video_msr_caption('/mnt/tags/msr-caption-all', video_name)
        _rcnn_data = load_video_rcnn('/mnt/tags/rcnn-info-all', video_name)
        
        # compose video term freq (a list of dicts)
        frame_names = os.listdir(os.path.join('/mnt/frames', video_name))
        frame_names = sorted(frame_names, key= lambda x: int(x.split('.')[0]))
        print 'start getting tfs for all frames'
        all_tfs = naive_combine_models(video_name, frame_names, _vgg_data, _msr_cap_data, _rcnn_data) 
        print 'Done loading tfs for all frames'
        
        # uniformly subsample frames 
        FRAME_RETAIN_RATE = 10/100. 
        for frame_rate in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            retained_frames = naive_subsample_frames(video_name, frame_rate)
            print 'Done loading tfs for subsampled frames'
            subsampled_tfs = naive_combine_models(video_name, retained_frames, _vgg_data, _msr_cap_data, _rcnn_data) 
       
        
            print frame_rate, detailed_measure(all_tfs, subsampled_tfs) 
        break 
