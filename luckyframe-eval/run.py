import sys
TOOL_PATH = '/home/t-yuche/clustering/tools'
sys.path.append(TOOL_PATH)
from utils import *


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


def naive_skipframes(video_name, FRAME_RETAIN_RATE):

    frame_names = os.listdir(os.path.join('/mnt/frames', video_name))
    frame_names = sorted(frame_names, key= lambda x: int(x.split('.')[0]))

    n_picked_frames = len(frame_names) * FRAME_RETAIN_RATE
    step = n_picked_frames/((len(frame_names) ) * 1.0)
    track = 0
    counter = 0
    retained_frames = [] 
    for idx, frame_name in enumerate(frame_names):
        if int(track) == counter:
            print 'keep - ', frame_name
            retained_frames += [frame_name]
            counter += 1
        print 'skip', frame_name
        track += step

    print len(retained_frames)/(len(frame_names) * 1.0)
    return retained_frames, len(frame_names)/(len(retained_frames) * 1.0)

#def detail_measure(): 
'''
# of subsampled words/# of all words
'''


#def hist_measure():
'''
similarity of the word distribution
'''

#def time_dist_measure():
'''
'''

#def combine_cnns():
    
     

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
    



if __name__ == "__main__":
   
    VIDEO_LIST = '/mnt/video_list.txt'
    videos = open(VIDEO_LIST).read().split()

    for video_name in videos:
        print video_name
         
        _rcnn_data = load_video_rcnn('/mnt/tags/rcnn-info-all', video_name)
        #_vgg_data = load_video_recog('/mnt/tags/vgg-classify-all', video_name)
        #_fei_caption_data = load_video_caption('/mnt/tags/fei-caption-all', video_name)
        #_msr_cap_data = load_video_msr_caption('/mnt/tags/msr-caption-all', video_name)

        FRAME_RETAIN_RATE = 10/100. 
        naive_skipframes(video_name, FRAME_RETAIN_RATE)  
        exit(-1) 
