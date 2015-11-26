import random
from utils import *
import pickle
import numpy as np
SERVER_STORAGE_FRAMES = 5 * 30

def get_inrange_fids(start_fid, end_fid, subsampled_fids):

    in_range_fids = []
    for f_count, fid in enumerate(subsampled_fids):
               
        if fid >= start_fid and fid < end_fid:
            in_range_fids += [fid]
 
    if len(in_range_fids) == 0:
        for f_count, fid in enumerate(subsampled_fids):
            
            if f_count == len(subsampled_fids)-1:
                if fid < start_fid:
                    in_range_fids += [fid]
                    break
            elif fid < start_fid and subsampled_fids[f_count + 1] >= end_fid:
                    in_range_fids += [fid]
                    break

    return in_range_fids 


def get_videoseg_name(video_name, fid):

    n_frames = get_video_frame_num(video_name)
   
    chunk = fid / SERVER_STORAGE_FRAMES 
    start_fid = chunk * SERVER_STORAGE_FRAMES
    end_fid = start_fid + SERVER_STORAGE_FRAMES

    videoseg_name = video_name + '_' + str(start_fid) + '_' + str(end_fid)  + '.mp4'
 
    return videoseg_name, start_fid, end_fid

def get_center(bbox):
   
    return ((bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2)
    
def get_dist(x, y, w, h):
    return math.sqrt( ((x[0] - y[0])/(w * 1.0))** 2 + ((x[1]-y[1])/(h * 1.0)) ** 2)


def get_center_dist(a, b, w, h):
    a_c = get_center(a)
    b_c = get_center(b)
    
    return get_dist(a_c, b_c, w,h)

def get_nearest_bbox(target_bbx, bboxes, w, h):

    if len(bboxes) == 1:
        return bboxes

    min_dist = 1000
    near_bbox = None 

    for bbox in bboxes:
        dist = get_center_dist(target_bbx, bbox, w,h)
        if dist < min_dist:
            min_dist = dist
            near_bbox = bbox
        
    return near_bbox    

def get_obj_size(bbox):
    bbx_w = (bbox[2] - bbox[0])
    bbx_h = (bbox[3] - bbox[1])

    return bbx_w * bbx_h

    

def get_biggest_obj(objs): 

    if len(objs) == 1:
        return objs[0]

    max_size = 0
    biggest_obj = None

    for obj in objs:
        s = get_obj_size(obj)
        if s > max_size:
            max_size = s
            biggest_obj = obj
    

    return biggest_obj

def process_visual_hint(chunk, w, h):

    CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    obj_trace = {}
    ## features: number of objs, dwell time, moving speed, obj size
    ## init
    for CLASS in CLASSES:
        obj_trace[CLASS] = {'trace': [], 'dwell_time': 0, 'obj_num': 0, 'moving_speed': -1, 'ave_obj_size': 0, 'max_obj_size': 0 }

    for CLASS in CLASSES:
        
        dwell_time_count = 0
        obj_count = 0
        ave_size = (0, 0)
        max_size = (0, 0) 
        target_bbox = None
        speeds = []
        prev_fid = -1
        for idx, frame_info in enumerate(chunk):
            objs = []
            for obj in frame_info['pred']:
                if obj['score'] >= 0.5 and obj['class'] == CLASS:
                    if len(objs) == 0: 
                        dwell_time_count += 1

                    bbx = obj['bbox']
                    bbx_w = (bbox[2] - bbox[0])/ (w * 1.0)
                    bbx_h = (bbox[3] - bbox[1])/ (h * 1.0)
                    obj_size = ( bbox_w, bbox_h )
                    

                    if obj_size[0] * obj_size[1]:
                        max_size[0] = obj_size[0] 
                        max_size[1] = obj_size[1]
 
                    ave_size[0] += obj_size[0]
                    ave_size[1] += obj_size[1]
                    objs += [bbx]   
                    obj_count += 1 

            if len(obj_trace[CLASS]['trace']) == 0 and len(objs) > 0:
                target_bbox = get_biggest_obj(objs)
                prev_fid = idx
            elif target_bbx not None and len(objs) > 0:
                nbbx = get_nearest_bbox(target_bbx, objs, w,h) 
                d = get_center_dist(target_bbx, nbbx, w, h)
                speeds += [d/((idx - prev_fid) * 1.0)]  
                prev_fid = idx 

            obj_trace[CLASS]['trace'] += [objs]


        obj_trace[CLASS]['dwell_time'] = dwell_time_count
        obj_trace[CLASS]['obj_num'] = obj_count/len(chunk)
        obj_trace[CLASS]['moving_speed'] = np.mean(speeds) 
        obj_trace[CLASS]['ave_obj_size'] = (ave_size[0]/obj_count, ave_size[1]/obj_count)
        obj_trace[CLASS]['max_obj_size'] = (max_size[0], max_size[1])

if __name__ == "__main__":

    videos = open('/mnt/video_list.txt').read().split() 
    #OPTIMAL_INPUT_FOLDER = '/home/t-yuche/admission-control/eval/optimal-log/optimal-tf'
    OPTIMAL_INPUT_FOLDER = '/home/t-yuche/admission-control/eval/optimal-log/optimal-tf-less-stopword'
    GREEDYFOLDER = '/home/t-yuche/admission-control/greedy/window-greedy-log-0.5'

    '''
    start_fid_set = []
    video_length = {}
    for i in xrange(50):
        random.seed(i)
        start_fid = {}
        for vid, video_name in enumerate(videos):

            if video_name not in video_length.keys():
                n_frames = get_video_frame_num(video_name)
                video_length[video_name] = n_frames

            n_frames = video_length[video_name]
            print i, vid
            start_fid[video_name] = random.randint(0, n_frames) 
        start_fid_set += [start_fid]
       
    with open('start_fid_set.pickle', 'wb') as fh:
        pickle.dump(start_fid_set, fh)       
    '''


    with open('start_fid_set.pickle') as fh:
        start_fid_set = pickle.load(fh) 
 
    for ss in start_fid_set:
        for video_name in videos: 
            
            n_frames = get_video_frame_num(video_name)
            rcnn_dict, vgg_dict, dummy, msr_cap_dict, dummy = load_all_modules_dict(video_name)

            with open(os.path.join(OPTIMAL_INPUT_FOLDER, video_name + '_0.5.pickle')) as fh:
                optimal_data = pickle.load(fh)

            # populate text and visual features
            video_seg_name, start_fid, end_fid = get_videoseg_name(video_name, ss[video_name])
            end_fid = min(start_fid + SERVER_STORAGE_FRAMES, n_frames)
            key = str(start_fid) + '-' + str(end_fid)

            greedypath = os.path.join(GREEDYFOLDER, video_name + '_0.5_gtframe.pickle')
            gt_data = pickle.load(open(greedypath))
            gt_picked_fid = gt_data['picked_f']
            total_frame_n = gt_data['total_frame']

            # text features from greedy 
            inrange_fids = get_inrange_fids(start_fid, end_fid, gt_picked_fid)
            inrange_frames = [os.path.join('/mnt/frames', video_name, str(x) + '.jpg') for x in inrange_fids]   
    
            tf_list = combine_all_modeldicts(vgg_dict, msr_cap_dict, rcnn_dict, dummy, inrange_frames, 1)
            tf = get_combined_tfs(tf_list)

            # visual hints
            rcnn_bbx_list, rcnn_bbx_dict = load_video_rcnn_bbx('/mnt/tags/rcnn-bbx-all', video_name)
 
            ''' subsampled (greedy) + visual hint'''
            # use tf + visual hints 
              
            ''' subsampled (greedy) '''
            # use tf          

            ''' subsampled (ml) + visual hint '''
            # TODO 
            ''' subsampled (ml) '''
            # TODO
            ''' all + visual hint'''
            optimal_tf = optimal_data[key]
            video_chunk = rcnn_bbx_list[start_idx: end_fid] 
             


            ''' metadata '''
