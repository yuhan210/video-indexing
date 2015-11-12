from utils import *
import numpy as np
import math

def get_obj_count(chunk, OBJ_NAME, THRESH =0.5):


    obj_counts = []    
    for frame_info in chunk:
    
        obj_count = 0
        for obj in frame_info['pred']:
            if obj['score'] < THRESH:
                continue
            if obj['class'] in OBJ_NAME:
                obj_count += 1
        
        obj_counts += [obj_count]
 

    return obj_counts

     

def check_video_chunk(chunk, w, h):
    #OBJ_NAME = ['dog', 'horse', 'cat', 'bird']
    OBJ_NAME = ['dog']
    chunk_info = {}
    counts = get_obj_count(chunk, OBJ_NAME)

    #print counts
    for count in counts:
        if count > 1:
            return False, chunk_info

    if sum(counts) < 1: 
        return False, chunk_info


    chunk_info['dwell_time'] = sum(counts) 
    chunk_info['obj_count']  = 1


    first_obj = None
    last_obj = None
    obj_sizes = []
    for idx, item in enumerate(chunk):
        # for each frame
        for obj in item['pred']: 
            # for each prediction 
            if obj['class'] in OBJ_NAME:
                bbox = obj['bbox']
                bbox_w = bbox[2] - bbox[0]
                bbox_h = bbox[3] - bbox[1]
                obj_sizes += [( bbox_w/(w * 1.0), bbox_h/(h * 1.0) )] 
                if first_obj == None:
                    first_obj = obj
                last_obj = obj

    chunk_info['obj_size'] =  (np.mean([x[0] for x in obj_sizes]), np.mean([x[1] for x in obj_sizes]))

    longest_path = math.sqrt(w ** 2 + h ** 2)
    chunk_info['moving_lvl'] = math.sqrt((last_obj['bbox'][0] - first_obj['bbox'][0])**2 + (last_obj['bbox'][1] - first_obj['bbox'][1]) ** 2)/(longest_path * 1.0)

    return True, chunk_info

if __name__ == "__main__":

    videos = open('/mnt/video_list.txt').read().split()
    #videos = open('video_list_tmp').read().split()
    WINDOW_SIZE = 5 * 30 # 5 secs, 30 fps

    outpath = '/home/t-yuche/visual-hint/video-seg-info/single_obj_tmp.txt'
    outf = open(outpath, 'w')

    for idx, video_name in enumerate(videos):
        print video_name
        rcnn_bbx_list, rcnn_bbx_dict = load_video_rcnn_bbx('/mnt/tags/rcnn-bbx-all', video_name) 
        w, h = get_video_res(video_name)       
 
        video_len = len(rcnn_bbx_list)
        start_idx = 0
        while start_idx + WINDOW_SIZE < video_len:
            video_chunk = [] 
            video_chunk = rcnn_bbx_list[start_idx: start_idx + WINDOW_SIZE]
            status, chunk_info = check_video_chunk(video_chunk, w, h) 

            #print chunk_info
            if len(chunk_info) > 0:
                output_line = video_name + ',' + str(start_idx)  + ',' + str(start_idx + WINDOW_SIZE -1) + ',' + str(chunk_info['obj_count']) + ',' + str(chunk_info['dwell_time']) + ',' + str(chunk_info['obj_size']) + ',' + str(chunk_info['moving_lvl']) + '\n'
            
                outf.write(output_line) 
                outf.flush()
            start_idx += WINDOW_SIZE

    outf.close()
