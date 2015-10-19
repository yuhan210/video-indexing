from PIL import Image
from utils import *
import os
import pickle

if __name__ == "__main__":

    VIDEO = '/mnt/video_list.txt'

    videos = open(VIDEO).read().split()
    obj_detect_processingtime = {}
    msr_processingtime = []
    rcnn_processingtime = []
    for vid, video_name in enumerate(videos):
        print video_name

        frame_names = os.listdir(os.path.join('/mnt/frames', video_name))
        # edge box profiling
        # get image resolution
        filepath = os.path.join('/mnt/frames', video_name, frame_names[0])
        im = Image.open(filepath)
        (w, h) = im.size # (width,height) tuple
        resol = str(w) + 'x' + str(h)
        for frame_name in frame_names:
            fid = frame_name.split('.')[0]
            bbox_path = os.path.join('/mnt/tags/edgebox-all/', video_name, fid +'.bbx')
            with open(bbox_path, 'r') as f:
                time_line = float(f.readline().strip().split(' ')[-1])
                if resol not in obj_detect_processingtime:
                    obj_detect_processingtime[resol] = [time_line]
                else:  
                    obj_detect_processingtime[resol] += [time_line]

        # msr cap
        msr_cap_data, msr_cap_dict = load_video_msr_caption('/mnt/tags/msr-caption-all', video_name) 
        for frame_path in msr_cap_dict:
            msr_processingtime += [msr_cap_dict[frame_path]['caption_time']]

        # rcnn_processing time
        rcnn_data = load_video_rcnn_bbx('/mnt/tags/rcnn-bbx-all', video_name) 
        for item in rcnn_data:
            if 'rcnn_time' in item:
                rcnn_processingtime += [item['rcnn_time']] 

        if vid % 50 == 0:

            with open('rcnn_processing_time.pickle', 'wb') as f:
                pickle.dump(rcnn_processingtime, f)
            with open('msr_processing_time.pickle', 'wb') as f:
                pickle.dump(msr_processingtime, f)
            with open('obj_detection_processing_time.pickle', 'wb') as f:
                pickle.dump(obj_detect_processingtime, f)


    with open('rcnn_processing_time.pickle', 'wb') as f:
        pickle.dump(rcnn_processingtime, f)
    with open('msr_processing_time.pickle', 'wb') as f:
        pickle.dump(msr_processingtime, f)
    with open('obj_detection_processing_time.pickle', 'wb') as f:
        pickle.dump(obj_detect_processingtime, f)


