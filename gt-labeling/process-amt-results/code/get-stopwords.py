from utils import *
import pickle

if __name__ == "__main__":

    VIDEO_LIST = '/mnt/video_list.txt'
    videos = open(VIDEO_LIST).read().split()
    word_count = {}

    for vid, video_name in enumerate(videos):
        print vid, video_name
        rcnn_data, vgg_data, fei_data, msr_data, rcnn_bbx = load_all_modules(video_name) 
        turker_labels = load_video_processed_turker(video_name)

        if rcnn_data == None:
            continue
        

        tf_list = combine_all_models(video_name, vgg_data, msr_data, rcnn_data, fei_data) 

        for label_obj in turker_labels:
            frame_name = label_obj['frame_name']
            gt_labels = label_obj['gt_labels'] 
        
            combined_tf = filter(lambda x: x['frame_name'] == frame_name, tf_list)[0]['tf']
            
            for w in combined_tf:
                if w not in word_count:

                    if w in gt_labels:
                        word_count[w] = [1, 1]
                    else:
                        word_count[w] = [1, 0]

                else:
                    word_count[w][0] += 1
                    if w in gt_labels:
                        word_count[w][1] += 1

        
    with open('./all_word_count.txt', 'wb') as fh:
        pickle.dump(word_count, fh) 
