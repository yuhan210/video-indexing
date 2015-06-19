from nltk.stem.wordnet import WordNetLemmatizer
import json
import nltk
import os

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

