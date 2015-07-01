from tools.utils import *
from nltk.corpus import stopwords
import sys
import json
import os

def removeStopWords(list_str):
    ws = []
    for s in list_str:
       ws.extend( [w for w in s.split(' ') if w not in stopwords.words('english')] )

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
            words[w] = 5
        else:
            words[w] += 5

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


def getSuggestedChoices(rcnn_data, vgg_data, caption_data, start_frame, end_frame):

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
    

if __name__ == "__main__":

 
    video_list = [x.strip() for x in open('/home/t-yuche/deep-video/data/videos.txt').readlines()]

    # for each video
    for v in video_list:

        print v 
        anno_folder = os.path.join('/home/t-yuche/gt-labeling/frame-subsample/annos', v)
        if not os.path.exists(anno_folder):
            os.mkdir(anno_folder)
   
        rcnn_data = load_video_rcnn('/home/t-yuche/frame-analysis/rcnn-info', v)
        vgg_data = load_video_recog('/home/t-yuche/frame-analysis/recognition', v)
        caption_data = load_video_caption('/home/t-yuche/frame-analysis/caption', v)
 
        keyframes = json.load(open('./keyframe-info/' + v + '_uniform.json'))['img_blobs']

        total_kframes = len(keyframes)
        total_frames = len(rcnn_data)
      
        # process possible labels
        for idx, kf in enumerate(keyframes):
            kf_name = kf['key_frame']
            frame_idx = int(kf_name.split('.')[0])
            
            # get labels, merge labels in range
            labels = [] 
            if idx == total_kframes - 1:
                labels = getSuggestedChoices(rcnn_data, vgg_data, caption_data, keyframes[idx]['key_frame'], rcnn_data[len(rcnn_data)-1]['image_path'].split('/')[-1])
                
            else:
                labels = getSuggestedChoices(rcnn_data, vgg_data, caption_data, keyframes[idx]['key_frame'], keyframes[idx+1]['key_frame'])
 
            output_dict = {}
            label_dict = {}
            for idx, label in enumerate(labels[:20]):
                label_dict[str(idx)] = str(label[0])
            
            # sort label_dict
            output_dict['choices'] = label_dict
            anno_file = os.path.join(anno_folder, kf_name.split('.')[0] + '.json')
            json.dump(output_dict, open(anno_file, 'w'))
