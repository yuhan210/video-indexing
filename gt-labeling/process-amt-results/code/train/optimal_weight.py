from nltk.corpus import stopwords
import inflection 
import random
import sys
import json
import os
TOOL_PATH = '/home/t-yuche/clustering/tools'
sys.path.append(TOOL_PATH)
from utils import *

    

def gen_train_str(w, rcnn_ws, vgg_ws, msr_cap_ws, fei_cap_ws, turker_labels_list, STOPWORDS):

    w = inflection.singularize(w)
    if w in STOPWORDS:
        return None

     

if __name__ == "__main__":


    RCNN_IDX = 0
    VGG_IDX = 1
    FEI_IDX = 2
    MSR_IDX = 3

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
        
    STOPWORDS = get_stopwords(1)

    train_fh = open('train_log.txt', 'w')

    for video_name in videos:
        print video_name

        dummy, turker_labels = load_video_processed_turker(video_name)
        rcnn_dict, vgg_dict, fei_cap_dict, msr_cap_dict, dummy = load_all_modules_dict(video_name)    
 
        # process possible labels
        train_data = {}
        for idx, frame_name in enumerate(turker_labels):
            frame_path = os.path.join('/mnt/frames', video_name, frame_name) 

            for rcnn_idx, rcnn_w in enumerate(rcnn_dict[frame_path]['pred']['text']):
                conf = rcnn_dict[frame_path]['pred']['conf'][rcnn_idx]
                rcnn_w = inflection.singularize(rcnn_w)
                if rcnn_w in STOPWORDS:
                    continue      

                if rcnn_w not in train_data:
                    train_data[rcnn_w] = [conf, 0, 0, 0]
                else:
                    train_data[rcnn_w][RCNN_IDX] = conf

            for vgg_idx, vgg_w in enumerate(vgg_dict[frame_path]['pred']['text']):
                conf = vgg_dict[frame_path]['pred']['conf'][vgg_idx]
                vgg_w = inflection.singularize(vgg_w)
                if vgg_w in STOPWORDS:
                    continue

                if vgg_w not in train_data:
                    train_data[vgg_w] = [0, conf, 0, 0]
                else:
                    train_data[vgg_w][VGG_IDX] = conf
   
            
 
    train_fh.close()
