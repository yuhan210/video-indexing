from nltk.corpus import stopwords
import sys
import pickle
CLUSTER_PATH = '/home/t-yuche/clustering/clusterLib'
sys.path.append(CLUSTER_PATH)
from cluster import *
TOOL_PATH = '/home/t-yuche/clustering/tools'
sys.path.append(TOOL_PATH)
from utils import *

def removeStopWords(list_str):
    ws = []
    for s in list_str:
       ws.extend( [w for w in s.split(' ') if w not in stopwords.words('english')] )

    return ws


def removeStopWordsFromWordlist(list_ws):
    ws = []
    for s in list_ws:
        if s not in stopwords.words('english') and s.find('background') < 0:
            ws += [s]

    return ws

def getwnid(w):

    with open('synset_words.txt') as f:
        for l in f.readlines():
            wnid = l.strip().split(' ')[0]
            name = [x.strip() for x in ' '.join(l.strip().split(' ')[1:]).split(',')][0]

            if name == w:
                return wnid


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


def genBow(rcnn_ws, vgg_ws, fei_caption_ws, msr_caption_ws):

    words = {}
    for w in rcnn_ws:
        if w not in words:
            words[w] = 3
        else:
            words[w] += 3

    for w in vgg_ws:
        # compose vgg word
        w = composeVGGWordnet(w)
        if w.find('->') >= 0:
            w = w.split('->')[-1]
        if w not in words:
            words[w] = 3
        else:
            words[w] += 3

    for w in fei_caption_ws:
        if w not in words:
            words[w] = 1
        else:
            words[w] += 1

    for w in msr_caption_ws:
        if w not in words:
            words[w] = 3
        else:
            words[w] += 3

    return words



def composeNodes(keyframes, _rcnn_data, _vgg_data, _fei_cap_data, _msr_cap_data):

    nodes = []
    for frame_name in keyframes:
        start_idx = int(frame_name.split('.')[0])
        
        rcnn_data = filter(lambda x: int(x['image_path'].split('/')[-1].split('.')[0]) == start_idx, _rcnn_data)
        vgg_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == start_idx, _vgg_data)
        fei_cap_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == start_idx, _fei_cap_data)
        msr_cap_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == start_idx, _msr_cap_data)

        ## process fast-rcnn
        idx = 0

        rcnn_ws = []
        if len(rcnn_data) > 0:
            for rcnn_idx, pred in enumerate(rcnn_data[idx]['pred']['text']):
    
                ## the confidence is higher than 10^(-3) and is not background
                if rcnn_data[idx]['pred']['conf'][rcnn_idx] > 0.0005 and  pred.find('background') < 0:
                    rcnn_ws += [pred]

        ## process vgg
        vgg_ws = []
        if len(vgg_data) > 0:
            vgg_ws = [w for w in vgg_data[idx]['pred']['text']]

        ## process neuraltalk
        fei_cap_ws = []
        if len(fei_cap_data) > 0:
            fei_cap_ws = removeStopWords(fei_cap_data[idx]['candidate']['text'])

        ## process msr captioning
        msr_cap_ws = []
        if len(msr_cap_data) > 0:
            msr_cap_ws = removeStopWordsFromWordlist(msr_cap_data[idx]['words']['text'])
            msr_cap_data[idx]['words']['prob']

        words = {}
        for w in rcnn_ws:
            if w not in words:                                                                            
                words[w] = 3
            else:                                                                                         
                words[w] += 3                                                                             

        for w in vgg_ws:                                                                                  
        # compose vgg word
            w = composeVGGWordnet(w)
            if w.find('->') >= 0:                                                                         
                w = w.split('->')[-1]
            if w not in words:
                words[w] = 3                                                                              
            else:
                words[w] += 3                                                                             

        for w in fei_cap_ws:
            if w not in words:
                words[w] = 1
            else:                                                                                         
                words[w] += 1                                                                             
    
        for w_idx, w in enumerate(msr_cap_data[idx]['words']['text']):
            score = float(msr_cap_data[idx]['words']['prob'][w_idx])
            if w not in stopwords.words('english')  and w.find('background') < 0:
                if w not in words:
                    words[w] = max(1, 5 + score)
                else:
                    words[w] += max(1, 5 + score)
      
        # sort by timestamp
        nodes += [(start_idx, words)]
    nodes = sorted(nodes, key=lambda x:x[0])
    return nodes 

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print 'Usage:', sys.argv[0], ' (0: reset 1: append) index_file video_list'
        exit(-1)
    
    IS_APPEND = int(sys.argv[1])
    index_file = sys.argv[2]
    video_list_file = sys.argv[3]
    video_list = open(video_list_file).read().splitlines()

    index = {}
    if IS_APPEND:
        with open(index_file, 'rb') as handle:
            prev_index = pickle.load(handle)
            print prev_index    


    for v_idx, video_name in enumerate(video_list):
        video_name = video_name.split('.')[0]
        print video_name 
        if IS_APPEND:
            if video_name in index:
                continue

        # load keyframes
        keyframe_paths = loadKeyFrames(video_name)
        keyframes = loadKeyFrameFilenames(video_name)
        keyframes = [x['key_frame'] for x in keyframes]
        # construct nodes for clustering
        rcnn_data, vgg_data, fei_cap_data, msr_cap_data = load_all_labels(video_name) 
        nodes = composeNodes(keyframes, rcnn_data, vgg_data, fei_cap_data, msr_cap_data)
        
        # construct index 
        clusters, linkage_matrix = cluster(nodes)
        index[video_name] = clusters
        
        ## write checkpoint
        if v_idx % 50 == 0: 
            with open('./tmp/' + str(v_idx) + '_' + index_file, 'wb') as handle:
                pickle.dump(index, handle)
