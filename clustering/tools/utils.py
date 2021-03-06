from nltk.stem.wordnet import WordNetLemmatizer
import json
import nltk
import csv
import os


def loadKeyFrames(video_name):

    KEYFRAME_FOLDER = '/home/t-yuche/gt-labeling/frame-subsample/keyframe-info'
    keyframe_file = os.path.join(KEYFRAME_FOLDER, video_name + '_uniform.json')

    with open(keyframe_file) as json_file:
        keyframes = json.load(json_file)

    FRAME_FOLDER = '/mnt/frames'
    keyframe_filenames = [ os.path.join(FRAME_FOLDER, video_name, x['key_frame']) for x in keyframes['img_blobs'] ]

    return keyframe_filenames

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



def load_video_turker(turker_folder, video_name):
    '''
    Return a dict: 
    key: frame_id (str)
    value: a list of smaller lists(each list is a choice)
    '''
   
    folder = os.path.join(turker_folder, video_name)
    files = sorted(os.listdir(folder), key = lambda x: int(x.split('.')[0]))
    ds = {}
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

