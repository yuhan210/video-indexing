from nltk.stem.wordnet import WordNetLemmatizer
from nlp import *
from wordnet import *
import inflection 
import math
import json
import nltk
import csv
import os

def load_video_rcnn_bbx(rcnn_bbx_folder, video_name):

    file_pref = os.path.join(rcnn_bbx_folder, video_name)    

    # load rcnn bbx
    with open(file_pref + '_rcnnbbx.json') as json_file:
        rcnn_bbx_data = json.load(json_file)

    rcnn_bbx_data = sorted(rcnn_bbx_data['imgblobs'], key=lambda x: int(x['img_path'].split('/')[-1].split('.')[0]))

    return rcnn_bbx_data


def cos_similarty(a_dict, b_dict):
    '''
    Compute the cos similarity between two tfs (two dictionary)
    '''   
 
    space = list(set(a_dict.keys()) | set(b_dict.keys()))
    
    # compute consine similarity (a dot b/ ||a|| * ||b||)
    sumab = 0.0
    sumaa = 0.0
    sumbb = 0.0

    for dim in space:

        a = 0.0
        b = 0.0
        if dim in a_dict:
            a = a_dict[dim]
        if dim in b_dict:
            b = b_dict[dim]
        
        sumab += a * b
        sumaa += a * a
        sumbb += b * b        
    
    return sumab/(math.sqrt(sumaa) * math.sqrt(sumbb))
 
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

def combine_all_models(video_name, _vgg_data, _msr_data, _rcnn_data, _fei_data):

    stop_words = get_stopwords()
    wptospd = word_pref_to_stopword_pref_dict()

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
            vgg_ws = [w for w in vgg_data['pred']['text']]
    
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
                if w not in stop_words and w not in msr_ws:
                    msr_ws += [w]

        words = {}
        for w in rcnn_ws:
            if w not in words:
                words[w] = 1
            else:
                words[w] += 1
        for w in vgg_ws:
            w = wptospd[w] 
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

        tf_list += [{'frame_name': frame_name, 'tf': words}]

    return tf_list


def getwnid(w):

    with open('/home/t-yuche/caffe/data/ilsvrc12/synset_words.txt') as f:
        for l in f.readlines():
            wnid = l.strip().split(' ')[0]
            name = [x.strip() for x in ' '.join(l.strip().split(' ')[1:]).split(',')][0]

            if name == w:
                return wnid


def loadKeyFrames(video_name):

    KEYFRAME_FOLDER = '/home/t-yuche/gt-labeling/frame-subsample/keyframe-info'
    keyframe_file = os.path.join(KEYFRAME_FOLDER, video_name + '_uniform.json')

    with open(keyframe_file) as json_file:
        keyframes = json.load(json_file)

    FRAME_FOLDER = '/mnt/frames'
    keyframe_filenames = [ os.path.join(FRAME_FOLDER, video_name, x['key_frame']) for x in keyframes['img_blobs'] ]

    return keyframe_filenames

def load_all(video_name):
   
    if not os.path.exists(os.path.join('/mnt/tags/rcnn-info-all', video_name + '_rcnnrecog.json')) or not os.path.exists(os.path.join('/mnt/tags/vgg-classify-all', video_name + '_recog.json')) or not os.path.exists(os.path.join('/mnt/tags/msr-caption-all', video_name + '_msrcap.json')) or not os.path.exists(os.path.join('/mnt/tags/fei-caption-all', video_name + '_5_caption.json')) or not os.path.exists(os.path.join('/mnt/tags/rcnn-bbx-tmp', video_name + '_rcnnbbx.json')):
        return None, None, None, None, None

    rcnn_data = load_video_rcnn('/mnt/tags/rcnn-info-all', video_name)
    vgg_data = load_video_recog('/mnt/tags/vgg-classify-all', video_name)
    fei_caption_data = load_video_caption('/mnt/tags/fei-caption-all', video_name)
    msr_cap_data = load_video_msr_caption('/mnt/tags/msr-caption-all', video_name)
    rcnn_bbx = load_video_rcnn_bbx('/mnt/tags/rcnn-bbx-tmp', video_name) 

    return rcnn_data, vgg_data, fei_caption_data, msr_cap_data, rcnn_bbx
    

def load_all_labels(video_name):

    rcnn_data = load_video_rcnn('/mnt/tags/rcnn-info-all', video_name)
    vgg_data = load_video_recog('/mnt/tags/vgg-classify-keyframe', video_name)
    fei_caption_data = load_video_caption('/mnt/tags/fei-caption-keyframe', video_name)
    msr_cap_data = load_video_msr_caption('/mnt/tags/msr-caption-keyframe', video_name)

    return rcnn_data, vgg_data, fei_caption_data, msr_cap_data 
    

def loadKeyFrameFilenames(video_name):

    KEYFRAME_FOLDER = '/home/t-yuche/gt-labeling/frame-subsample/keyframe-info'
    keyframe_file = os.path.join(KEYFRAME_FOLDER, video_name + '_uniform.json')

    with open(keyframe_file) as json_file:
        keyframes = json.load(json_file)['img_blobs']

    return keyframes

def load_suggested_labels(video_name, anno_folder="/home/t-yuche/gt-labeling/suggested-labels"):

    files = os.listdir(os.path.join(anno_folder, video_name))
    files = sorted(files, key=lambda x: int(x.split('.')[0])) 
   
    ds = {}
    for f in files:
        with open(os.path.join(anno_folder, video_name, f)) as json_file:
            anno_data = json.load(json_file)  
            ds[f.split('.')[0] + '.jpg'] = anno_data['choices']
                     
    return ds



def load_video_processed_turker(turker_folder, video_name):
    '''
    Return singularized turker label
    '''
    file_path = os.path.join(turker_folder, video_name + '.json')
        
    if not os.path.exists(file_path):
        return None    

    with open(file_path) as fh:
        return json.load(fh) 

def load_video_turker(turker_folder, video_name):
    '''
    Return a dict: 
    key: frame_id (str)
    value: a list of smaller lists(each list is a choice)
    '''
    
    folder = os.path.join(turker_folder, video_name)
    
    ds = {}
    if not os.path.exists(folder):
        return ds

    files = sorted(os.listdir(folder), key = lambda x: int(x.split('.')[0]))
    for f in files:
        f_path = os.path.join(folder, f)
        with open(f_path) as json_file:
            turker_data = json.load(json_file)
            ds[f.split('.')[0] + '.jpg'] = turker_data['gt_labels'] 

    return ds

def load_turker_labels(amtresults_folder):
    
    ds = {}
    for f in os.listdir(amtresults_folder):
        csv_file = open(os.path.join(amtresults_folder, f))
        csv_reader = csv.DictReader(csv_file, delimiter="\t")
        
        for row in csv_reader:
            if row['Answer.n_selections'] != None and len(row['Answer.n_selections']) > 0:
                video_name = row['Answer.video']
                frame_name = row['Answer.frame_name']
                selections = row['Answer.selections'].split(',')
    
                if video_name not in ds:
                    ds[video_name] = []
                
                img_blob = {}
                img_blob['frame_name'] = frame_name
                img_blob['selections'] = selections
                ds[video_name] += [img_blob]
    
             
                #ds[video_name +  frame_name] = selections
        csv_file.close()
                 
    return ds            

def getVerbFromStr(sentence):

    segs = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(segs)
    
    verbs = []
    for word, tag in tags:
        if tag == 'VB' or tag == 'VBZ' or tag == 'VBN' or tag == 'VBD' or tag == 'VBG':
            verb_simple_tence = str(WordNetLemmatizer().lemmatize(word, 'v'))
            
            ## This is a hack. Remove 'be' verb  
            if verb_simple_tence != 'be': 
                verbs += [verb_simple_tence]

    return verbs


def getVerbTfFromCaps(captions, method = 'equal'):
    #TODO: implement weighted version
  
    verb_tf = {}
    for sentence in captions:

        verbs = getVerbFromStr(sentence)
        for verb in verbs:
            if verb not in verb_tf:
                verb_tf[verb] = 1
            else:
                verb_tf[verb] += 1        

    return verb_tf 

def load_video_msr_caption(msrcaption_folder, video_name):

    file_pref = os.path.join(msrcaption_folder, video_name)
    
    # load msr caption
    with open(file_pref + '_msrcap.json') as json_file:
        msrcap_data = json.load(json_file) 

    msrcap_data = sorted(msrcap_data['imgblobs'], key=lambda x: int(x['img_path'].split('/')[-1].split('.')[0]))

    return msrcap_data

def load_video_peopledet(peopled_folder, video_name):

    file_pref = os.path.join(peopled_folder, video_name)

    # load recognition
    with open(file_pref + '_openpd.json') as json_file:
        pd_data = json.load(json_file)

    pd_data = sorted(pd_data['img_blobs'], key=lambda x: int(x['img_name'].split('.')[0]))
    
    return pd_data


def load_video_ocr(ocr_folder, video_name):
    file_pref = os.path.join(ocr_folder, video_name)
    
    # load recognition
    with open(file_pref + '_ocr.json') as json_file:
        ocr_data = json.load(json_file)

    ocr_data = sorted(ocr_data['img_blobs'], key=lambda x: int(x['img_name'].split('.')[0]))
    
    return ocr_data

def load_video_dlibfd(dlibfd_folder, video_name):

    file_pref = os.path.join(dlibfd_folder, video_name)
    
    # load face detection
    with open(file_pref + '_dlibfd.json') as json_file:
        faced_data = json.load(json_file)

    faced_data = sorted(faced_data['img_blobs'], key=lambda x: int(x['img_name'].split('.')[0]))
    
    return faced_data

def load_video_rcnn(rcnn_folder, video_name):
    
    file_pref = os.path.join(rcnn_folder, video_name)
    
    # load face detection
    with open(file_pref + '_rcnnrecog.json') as json_file:
        rcnn_data = json.load(json_file)

    rcnn_data = sorted(rcnn_data['imgblobs'], key=lambda x: int(x['image_path'].split('/')[-1].split('.')[0]))
    
    return rcnn_data

    

def load_video_opencvfd(opencvfd_folder, video_name):
    file_pref = os.path.join(opencvfd_folder, video_name)
    
    # load face detection
    with open(file_pref + '_openfd.json') as json_file:
        faced_data = json.load(json_file)

    faced_data = sorted(faced_data['img_blobs'], key=lambda x: int(x['img_name'].split('.')[0]))
    
    return faced_data

def load_video_recog(recog_folder, video_name):
    
    file_pref = os.path.join(recog_folder, video_name)
    
    # load recognition
    with open(file_pref + '_recog.json') as json_file:
        recog_data = json.load(json_file)

    recog_data = sorted(recog_data['imgblobs'], key=lambda x: int(x['img_path'].split('/')[-1].split('.')[0]))
    
    return recog_data




def load_video_caption(caption_folder, video_name):

    file_pref = os.path.join(caption_folder, video_name)

    # load caption
    with open(file_pref + '_5_caption.json') as json_file:
        caption_data = json.load(json_file)

    # sort caption results based on frame number
    caption_data = sorted(caption_data['imgblobs'], key=lambda x:int(x['img_path'].split('/')[-1].split('.')[0]))

    return caption_data


def load_video_blur(blur_folder, video_name):

    file_pref = os.path.join(BLUR_folder, video_name)

    # load blurinfo
    with open(file_pref + '_blur.json') as json_file:
        blur_data = json.load(json_file)

    blur_data = sorted(blur_data['img_blobs'], key=lambda x:int(x['img_name'].split('.')[0]))

    return blur_data



def load_video_summary(summary_folder, video_name):

    file_pref = os.path.join(summary_folder, video_name)

    # load caption
    with open(file_pref + '_5_caption.json') as json_file:
        caption_data = json.load(json_file)

    # sort caption results based on frame number
    caption_data = sorted(caption_data['imgblobs'], key=lambda x:int(x['img_path'].split('/')[-1].split('.')[0]))

    # load recognition
    with open(file_pref + '_recog.json') as json_file:
        recog_data = json.load(json_file)

    recog_data = sorted(recog_data['imgblobs'], key=lambda x: int(x['img_path'].split('/')[-1].split('.')[0]))

    # load blurinfo
    with open(file_pref + '_blur.json') as json_file:
        blur_data = json.load(json_file)

    blur_data = sorted(blur_data['img_blobs'], key=lambda x:int(x['img_name'].split('.')[0]))

    return caption_data, recog_data, blur_data

