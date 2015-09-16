from nltk.corpus import stopwords
import random
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
   
 
def getSuggestedChoices(rcnn_data, vgg_data, fei_caption_data, msr_caption_data, frame_name, rcnn_w, vgg_w, fei_w, msr_w):

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
            words[w] = 4 * rcnn_w
        else:
            if 'rcnn' in word_src[w]:
                word_src[w]['rcnn'] += 4 
            else:
                word_src[w]['rcnn'] = 4 
            words[w] += 4 * rcnn_w

    for w in vgg_ws:
    # compose vgg word
        w = composeVGGWordnet(w)
        if w not in words:
            word_src[w] = {}
            word_src[w]['vgg']  = 4
            words[w] = 4 * vgg_w
        else:
            if 'vgg' in word_src[w]:
                word_src[w]['vgg'] += 4
            else:
                word_src[w]['vgg'] = 4
                
            words[w] += 4 * vgg_w

    for w in fei_caption_ws:
        if w not in words:
            word_src[w] = {}
            word_src[w]['fei']  = 1
            words[w] = 1 * fei_w
        else:
            if 'fei' in word_src[w]:
                word_src[w]['fei'] += 1
            else:
                word_src[w]['fei'] = 1
            words[w] += 1 * fei_w
    
    for w_idx, w in enumerate(msr_caption_data[idx]['words']['text']):
        score = float(msr_caption_data[idx]['words']['prob'][w_idx])
        if w not in stopwords.words('english') and w.find('background') < 0:
            if w not in words:
                word_src[w] = {}
                word_src[w]['msr']  = max(1, 5+ score)
                words[w] = max(1, 5 + score) * msr_w
            else:
                if 'msr' in word_src[w]:
                    word_src[w]['msr'] += max(1, 5 + score)
                else:
                    word_src[w]['msr'] = max(1, 5 + score)
                words[w] += max(1, 5 + score) * msr_w


    words = sorted(words.items(), key=lambda x: x[1], reverse=True)    
    labels = [x for x in words]
    return labels
    


if __name__ == "__main__":

    rcnn_ws = [0.05, 0.1, 0.2, 0.4]
    vgg_ws = [0.05, 0.1, 0.2, 0.4]
    fei_ws = [0.05, 0.1, 0.2, 0.3, 0.4]
    msr_ws = [0.05, 0.1, 0.2, 0.4, 0.5]

    SELECTED_VIDEOS = "sele_video_list.txt"
    if os.path.exists(SELECTED_VIDEOS):
        videos = open(SELECTED_VIDEOS).read().split()
    else:
        VIDEO_LIST = "/mnt/video_list.txt"
        videos = open(VIDEO_LIST).read().split()
        # randomly sample 250 videos
        videos = random.sample(videos, 250)    
        fh = open(SELECTED_VIDEOS, 'w')
        for video in videos:
            fh.write(video + '\n')

        fh.close() 

    result_fh = open('result.log', 'w')
    # fei, msr, rcnn, vgg 
    module_score = {'fei': 0, 'msr': 0, 'rcnn': 0, 'vgg': 0}

   
    for rcnn_w in rcnn_ws:
        for vgg_w in vgg_ws:
            for fei_w in fei_ws:
                for msr_w in msr_ws:
            
                    v_scores = []  
                    for v in videos:

                        print v
                        if not os.path.exists(os.path.join('/mnt/turker-labels', v)):
                            error_fh.write(v + '\n')
                            continue

                        ds = load_video_turker('/mnt/turker-labels', v)
                        anno_folder = os.path.join('/mnt/labels-for-turkers', v)

                        rcnn_data, vgg_data, fei_cap_data, msr_cap_data = load_all_labels(v)    

                        keyframes = loadKeyFrameFilenames(v)

                        total_kframes = len(keyframes)


                         
                        # process possible labels
                        f_score = []
                        for idx, kf in enumerate(keyframes):
                            kf_name = kf['key_frame']
                            frame_idx = int(kf_name.split('.')[0])
                            anno_file = os.path.join(anno_folder, kf_name.split('.')[0] + '.json')
                            
                            #labels, word_src = getSuggestedChoices(rcnn_data, vgg_data, fei_cap_data, msr_cap_data, kf_name)
                            labels = getSuggestedChoices(rcnn_data, vgg_data, fei_cap_data, msr_cap_data, kf_name, rcnn_w, vgg_w, fei_w, msr_w)
                            
                            labels = labels[:20]
                            labels = labels[::-1] 
                            labels = [x[0] for x in labels]
                           # print labels 

                            # weight module importance
                            amt_labels = ds[kf_name]
                            #print amt_labels
                            #print labels
                            total_score = 0
                            for _label in amt_labels:
                                score = 0
                                if len(_label) == 1: 
                                    if _label[0].split('-')[-1] in labels and _label[0].split('-')[-1] not in STOPWORDS:
                                        #print _label[0]
                                        label_str = _label[0].split('-')[-1]
                                        score = labels.index(label_str) + 1
                                else:
                                    if _label[-1].split('-')[-1] in labels and _label[-1].split('-')[-1] not in STOPWORDS:#vgg
                                        label_str = _label[-1].split('-')[-1]
                                        score = labels.index(label_str) + 1
                                 
                                total_score += score
                            f_score += [total_score/210.0] 
                        v_scores += [sum(f_score)/(len(keyframes) * 1.0)]
                        print sum(f_score)/(len(f_score) * 1.0)
                        print sum(f_score)/(len(keyframes) * 1.0)
                    result_fh.write(str(rcnn_w) + ',' +  str(vgg_w) +  str(fei_w) +  str(msr_w) +  str(sum(v_scores)/(len(v_scores) * 1.0)))
                    print rcnn_w, vgg_w, fei_w, msr_w, sum(v_scores)/(len(v_scores) * 1.0)


    result_fh.close()
