from nltk.corpus import stopwords
import sys
import json
import os
TOOL_PATH = '/home/t-yuche/clustering/tools'
sys.path.append(TOOL_PATH)
from utils import *


STOPWORDS = ['none','inside', 'near', 'two', 'day', 'front', u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not', u'only', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don', u'should', u'now']


def removeStopWordsFromWordlist(list_ws):

    ws = []
    for s in list_ws:
        if s not in stopwords.words('english') and s.find('background') < 0 and s not in STOPWORDS:
            ws += [s]

    return ws

def removeStopWords(list_str):
    ws = []
    for s in list_str:
       ws.extend( [w for w in s.split(' ') if w not in stopwords.words('english') and w not in STOPWORDS] )

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

def genBow(rcnn_ws, vgg_ws, caption_ws):

    words = {}
    for w in rcnn_ws: 
        if w not in words:
            words[w] = 3
        else:
            words[w] += 3

    for w in vgg_ws:
        # compose vgg word
        w = composeVGGWordnet(w)
        if w not in words:
            words[w] = 3
        else:
            words[w] += 3

    for w in caption_ws:
        if w not in words:
            words[w] = 1
        else:
            words[w] += 1

    return words

def getSuggestedLabel(rcnn_data, vgg_data, caption_data, start_frame, end_frame):

    start_idx = int(start_frame.split('.')[0])
    end_idx = int(end_frame.split('.')[0])

    rcnn_data = filter(lambda x: int(x['image_path'].split('/')[-1].split('.')[0]) >= start_idx and int(x['image_path'].split('/')[-1].split('.')[0]) <= end_idx , rcnn_data) 
    vgg_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) >= start_idx and int(x['img_path'].split('/')[-1].split('.')[0]) <= end_idx , vgg_data) 
    caption_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) >= start_idx and int(x['img_path'].split('/')[-1].split('.')[0]) <= end_idx , caption_data) 
  
    labels = []
    range_bows = {}
    for idx in xrange(len(rcnn_data)):
        
        rcnn_ws = []
        for rcnn_idx, pred in enumerate(rcnn_data[idx]['pred']['text']):
            ## the confidence is higher than 10^(-3) and is not background
            if rcnn_data[idx]['pred']['conf'][rcnn_idx] > 0.0005 and  pred.find('background') < 0:
                rcnn_ws += [pred]
                
        #rcnn_ws = [w for w in rcnn_data[idx]['pred']['text'] if w.find('background') < 0]
        vgg_ws = [w for w in vgg_data[idx]['pred']['text']]
        caption_ws = removeStopWords( caption_data[idx]['candidate']['text'] )

        bow = genBow(rcnn_ws, vgg_ws, caption_ws) 
        for w in bow:
            if w not in range_bows:
                range_bows[w] = 1
            else:
                range_bows[w] += 1

    range_bows = sorted(range_bows.items(), key=lambda x: x[1], reverse=True)    
    labels = [x for x in range_bows]
               
    return labels 




def getSuggestedChoices(rcnn_data, vgg_data, caption_data, start_frame):

    start_idx = int(start_frame.split('.')[0])

    rcnn_data = filter(lambda x: int(x['image_path'].split('/')[-1].split('.')[0]) == start_idx, rcnn_data) 
    vgg_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == start_idx, vgg_data) 
    caption_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == start_idx, caption_data) 
  
    labels = []
    range_bows = {}
    for idx in xrange(len(rcnn_data)):

        rcnn_ws = []
        for rcnn_idx, pred in enumerate(rcnn_data[idx]['pred']['text']):
            ## the confidence is higher than 10^(-3) and is not background
            if rcnn_data[idx]['pred']['conf'][rcnn_idx] > 0.0005 and  pred.find('background') < 0:
                rcnn_ws += [pred]
                
        #rcnn_ws = [w for w in rcnn_data[idx]['pred']['text'] if w.find('background') < 0]
        vgg_ws = [w for w in vgg_data[idx]['pred']['text']]
         
        caption_ws = removeStopWords( caption_data[idx]['candidate']['text'] )

        bow = genBow(rcnn_ws, vgg_ws, caption_ws) 
        for w in bow:
            if w not in range_bows:
                range_bows[w] = 1
            else:
                range_bows[w] += 1

    range_bows = sorted(range_bows.items(), key=lambda x: x[1], reverse=True)    
    labels = [x for x in range_bows]
    return labels 
    

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
    word_src = {}
    for w in rcnn_ws:
        if w not in words:
            word_src[w] = {}
            word_src[w]['rcnn'] = 4 
            words[w] = 4
        else:
            if 'rcnn' in word_src[w]:
                word_src[w]['rcnn'] += 4 
            else:
                word_src[w]['rcnn'] = 4 
            words[w] += 4

    for w in vgg_ws:
    # compose vgg word
        w = composeVGGWordnet(w)
        if w not in words:
            word_src[w] = {}
            word_src[w]['vgg']  = 4
            words[w] = 4
        else:
            if 'vgg' in word_src[w]:
                word_src[w]['vgg'] += 4
            else:
                word_src[w]['vgg'] = 4
                
            words[w] += 4

    for w in fei_caption_ws:
        if w not in words:
            word_src[w] = {}
            word_src[w]['fei']  = 1
            words[w] = 1
        else:
            if 'fei' in word_src[w]:
                word_src[w]['fei'] += 1
            else:
                word_src[w]['fei'] = 1
            words[w] += 1
    
    for w_idx, w in enumerate(msr_caption_data[idx]['words']['text']):
        score = float(msr_caption_data[idx]['words']['prob'][w_idx])
        if w not in stopwords.words('english') and w.find('background') < 0:
            if w not in words:
                word_src[w] = {}
                word_src[w]['msr']  = max(1, 5+ score)
                words[w] = max(1, 5 + score)
            else:
                if 'msr' in word_src[w]:
                    word_src[w]['msr'] += max(1, 5 + score)
                else:
                    word_src[w]['msr'] = max(1, 5 + score)
                words[w] += max(1, 5 + score)


    words = sorted(words.items(), key=lambda x: x[1], reverse=True)    
    labels = [x for x in words]
    return labels, word_src 
    

if __name__ == "__main__":

    VIDEO_LIST = "/mnt/video_list.txt"
    videos = open(VIDEO_LIST).read().split()
    
    # fei, msr, rcnn, vgg 
    module_score = {'fei': 0, 'msr': 0, 'rcnn': 0, 'vgg': 0}
    for v in videos:

        print v 
        ds = load_video_turker('/mnt/turker-labels', v)
        anno_folder = os.path.join('/mnt/labels-for-turkers', v)

        rcnn_data, vgg_data, fei_cap_data, msr_cap_data = load_all_labels(v)    

        keyframes = loadKeyFrameFilenames(v)

        total_kframes = len(keyframes)
      
        # process possible labels
        for idx, kf in enumerate(keyframes):
            kf_name = kf['key_frame']
            frame_idx = int(kf_name.split('.')[0])
            anno_file = os.path.join(anno_folder, kf_name.split('.')[0] + '.json')
            
            # get labels
            labels = []
             
            labels, word_src = getSuggestedChoices(rcnn_data, vgg_data, fei_cap_data, msr_cap_data, kf_name)
        
            output_dict = {}
            label_dict = {}
            sorted_w_src = []
            for idx, label in enumerate(labels[:20]):
                label_dict[str(idx)] = str(label[0])
                sorted_w_src += [{'wd': str(label[0]), 'src':word_src[str(label[0])] }] 

            # weight module importance
            amt_labels = ds[kf_name]
            for _label in amt_labels:
                if len(_label) == 1 and _label[0].split('-')[-1] in word_src and _label[0].split('-')[-1] not in STOPWORDS:
                    #print _label[0]
                    label_str = _label[0].split('-')[-1]
                    for key in word_src[label_str].keys():
                        module_score[key] += 1
                elif _label[0].split('-')[-1] not in STOPWORDS:#vgg
                    module_score['vgg'] += 1


    print module_score
    # normalize
    deno = module_score['fei'] + module_score['msr'] + module_score['rcnn'] + module_score['vgg']
    print 'fei', module_score['fei']/(demo * 1.0)
    print 'rcnn', module_score['rcnn']/(demo * 1.0)
    print 'msr', module_score['msr']/(demo * 1.0)
    print 'vgg', module_score['vgg']/(demo * 1.0)

         
         
