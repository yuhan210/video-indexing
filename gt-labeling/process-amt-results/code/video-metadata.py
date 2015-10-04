from utils import *
import pickle

if __name__ == "__main__":

    VIDEO_LIST = '/mnt/video_list.txt'
    METADATA_FOLDER = '/mnt/tags/video-metadata'
    videos = open(VIDEO_LIST).read().split()

    for vid, video_name in enumerate(videos):
        print vid, video_name
        rcnn_data, vgg_data, fei_data, msr_data, rcnn_bbx = load_all_modules(video_name) 
        #turker_labels = load_video_processed_turker(video_name)
        outfile_path = os.path.join(METADATA_FOLDER, video_name + '_metadata.pickle')

        if rcnn_data == None:
            continue
        
        tf_list = combine_all_models(video_name, vgg_data, msr_data, rcnn_data, fei_data) 
        agg_tf = {}
        for img_obj in tf_list:
            frame_name = img_obj['frame_name']
            tf = img_obj['tf']

            for w in tf:
                if w not in agg_tf:
                    agg_tf[w] = 1
                else:
                    agg_tf[w] += 1
            

        
        with open(outfile_path, 'wb') as fh:
            pickle.dump(agg_tf, fh) 
