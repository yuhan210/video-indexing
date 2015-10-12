from nltk.corpus import stopwords
import inflection 
import random
import sys
import json
import os
TOOL_PATH = '/home/t-yuche/clustering/tools'
sys.path.append(TOOL_PATH)
from utils import *

    


if __name__ == "__main__":


    RCNN_IDX = 0
    VGG_IDX = 1
    FEI_IDX = 2
    MSR_IDX = 3
    TURKER_IDX = 4

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
    wptospd = word_pref_to_stopword_pref_dict()
    convert_dict = convert_to_equal_word()

    train_fh = open('train_log.txt', 'w')
    for video_name in videos:
        print video_name

        dummy, turker_labels = load_video_processed_turker(video_name)
        rcnn_dict, vgg_dict, fei_cap_dict, msr_cap_dict, dummy = load_all_modules_dict(video_name)    
 
        # process possible labels
        for idx, frame_name in enumerate(turker_labels):
            train_data = {}
            frame_path = os.path.join('/mnt/frames', video_name, frame_name) 

            for rcnn_idx, rcnn_w in enumerate(rcnn_dict[frame_path]['pred']['text']):
                conf = rcnn_dict[frame_path]['pred']['conf'][rcnn_idx]
                rcnn_w = inflection.singularize(rcnn_w)
                if rcnn_w in STOPWORDS:
                    continue      

                if rcnn_w not in train_data:
                    train_data[rcnn_w] = [conf, 0, 0, 0, 0]
                else:
                    train_data[rcnn_w][RCNN_IDX] = conf

            for vgg_idx, vgg_w in enumerate(vgg_dict[frame_path]['pred']['text']):
                conf = vgg_dict[frame_path]['pred']['conf'][vgg_idx]
                vgg_w = wptospd[vgg_w]
                if vgg_w in convert_dict:
                    vgg_w = convert_dict[vgg_w]      
                vgg_w = inflection.singularize(vgg_w)
                 
                if vgg_w in STOPWORDS:
                    continue

                if vgg_w not in train_data:
                    train_data[vgg_w] = [0, conf, 0, 0, 0]
                else:
                    train_data[vgg_w][VGG_IDX] = conf
  
            for msr_idx, msr_w in enumerate(msr_cap_dict[frame_path]['words']['text']):
                msr_w = inflection.singularize(msr_w)
            
                if msr_w in STOPWORDS or len(msr_w) == 0:
                    continue

                conf = msr_cap_dict[frame_path]['words']['prob'][msr_idx]
                if msr_w not in train_data:
                    train_data[msr_w] = [0, 0, 0, conf, 0 ] 
                else:
                    train_data[msr_w][MSR_IDX] = conf


            for fei_idx, fei_s in enumerate(fei_cap_dict[frame_path]['candidate']['text']):
                conf = fei_cap_dict[frame_path]['candidate']['logprob'][fei_idx]
                for fei_w in fei_s.split(' '):
                    if len(fei_w) == 0 or fei_w in STOPWORDS:
                        continue
                    fei_w = inflection.singularize(fei_w)
                    if fei_w not in train_data:
                        train_data[fei_w] = [0, 0, conf, 0, 0]
                    else:
                        train_data[fei_w][FEI_IDX] = conf

            for label_w in turker_labels[frame_name]:
                if label_w in convert_dict:
                    label_w = convert_dict[label_w]

                if label_w in train_data:
                    train_data[label_w][TURKER_IDX] = 1

            for w in train_data:
                train_fh.write(w + ' ' + str(train_data[w][0]) + ' ' + str(train_data[w][1]) + ' ' + str(train_data[w][2]) + ' ' + str(train_data[w][3]) + ' ' + str(train_data[w][4]) + '\n')


    train_fh.close()
