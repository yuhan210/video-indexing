from utils import *
import pickle
import json
import random
import numpy as np 
#OPTIMAL_INPUT_FOLDER = '/home/t-yuche/admission-control/eval/optimal-log/optimal-tf'
OPTIMAL_INPUT_FOLDER = '/home/t-yuche/admission-control/eval/optimal-log/optimal-tf-less-stopword'
UNIMPORTANTWORD_THRESH = 0.5

SUBSAMPLE_INTPUT_FOLDER = '/home/t-yuche/admission-control/eval/greedy-results'

METADATA_FOLDER = '/mnt/video-info'
INDEX_FOLDER = '/home/t-yuche/ranking/gen_rank_data/optimal-index'
SERVER_WINDOW_SIZE = 5 * 30
SLIDING_SIZE = 1 * 30 
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


OUTPUT_FOLDER = 'label_videos'
def get_video_seg(rand_fid, video_name):

    start_fid = (rand_fid/SLIDING_SIZE) * SLIDING_SIZE
    end_fid = min(start_fid + SERVER_WINDOW_SIZE , video_framenum[video_name])

    return start_fid, end_fid

def get_type_score(value, value_type, TYPE = 0):

    if value_type == 'viewcount': 
        return value
    elif value_type == 'rating': 
        return 0
    elif value_type == 'dislikes':
        return 0
    elif value_type == 'likes': 
        return 0
    

def compute_metadata_score(stream_info, query_str): 
    TYPE = 1

    metadata_features = {'title': 0, 'rating': -1, 'description': 0, 'viewcount': -1, 'dislikes':-1, 'likes': -1,  'keywords': 0}
    query_words = query_str.split()

    # 
    words = stream_info['ftitle'].split('_')
    for q_w in query_words:
        for w in words:
            w = inflection.singularize(w.lower())
            if w.find(q_w) >= 0:
                metadata_features['title'] += 1
                break
    #
    metadata_features['rating'] = get_type_score(stream_info['rating'], 'rating', TYPE)
    
    #
    words = stream_info['description'].split()
    for q_w in query_words:
        for w in words:
            w = inflection.singularize(w.lower())
            if w.find(q_w) >= 0:
                metadata_features['description'] += 1
                break
    #
    metadata_features['viewcount'] = get_type_score(stream_info['viewcount'], 'viewcount', TYPE)
    #
    metadata_features['dislikes'] = get_type_score(stream_info['dislikes'], 'dislikes', TYPE)
    #
    metadata_features['likes'] = get_type_score(stream_info['likes'], 'likes', TYPE)

    #
    for q_w in query_words:        
        for w in stream_info['keywords']:
            w = inflection.singularize(w.lower())
            if w.find(q_w) >= 0:
                metadata_features['keywords'] += 1
                break
    
    # compute score
    score = 0
    if metadata_features['title'] or metadata_features['description'] or metadata_features['keywords']: 
        score = metadata_features['title'] + metadata_features['description'] + metadata_features['keywords']
        

    return score, metadata_features['viewcount']


def has_obj_using_gtlabel(video_name, start_fid, end_fid, query_str):

   
    all_labels = []
    n_labeled_frames = 0
    for fid in xrange(start_fid, end_fid + 1): 
        frame_name = str(fid) + '.jpg' 
        if frame_name in label_dict[video_name].keys():
            n_labeled_frames += 1
        
            gt_labels = label_dict[video_name][frame_name]
            all_labels += gt_labels 

    query_words = query_str.split() 
    match_count = 0
    non_match_count = 0
    for w in query_words:
        if n_labeled_frames > 0:
            match_portion = len(filter(lambda x: x == w, all_labels))/(n_labeled_frames* 1.0)
            if match_portion > 0.6:
                match_count += 1
            if match_portion == 0:
                non_match_count += 1

    return match_count, non_match_count


def getCosSimilarty(a_dict, b_dict):
    #print a_dict, b_dict
    space = list(set(a_dict.keys()) | set(b_dict.keys()))
    # compute consine similarity (a dot b/ |a| * |b|)
    sumab = 0.0
    sumaa = 0.0
    sumbb = 0.0

    for dim in space:
        a = 0.0
        b = 0.0
        if dim in a_dict.keys():
            a = a_dict[dim]
        if dim in b_dict.keys():
            b = b_dict[dim]

        sumab += a * b
        sumaa += a ** 2
        sumbb += b ** 2       
    
    if sumaa == 0 or sumbb == 0:
        return 0

    return sumab/(math.sqrt(sumaa * sumbb))



if __name__ == "__main__":

    global video_framenum
    with open('./video_frame_num.pickle') as fh:
        video_framenum = pickle.load(fh)

    with open('/home/t-yuche/ranking/gen_rank_data/start_fid_set.pickle') as fh:
        start_fid_set = pickle.load(fh)

    with open('./metadata_key_results.pickle') as fh:
        metadata_results = pickle.load(fh) 

    blocking_videos = {}
    with open('./blocking-videos') as fh:
        for line in fh.readlines():
            segs = line.strip().split()
            if segs[0] not in blocking_videos.keys():
                blocking_videos[segs[0]] = [segs[1]]
            else:
                blocking_videos[segs[0]] += [segs[1]]

    VIDEOS = open('/mnt/video_list.txt').read().split()

    global label_dict
    label_dict = {}
    global optimal_data
    optimal_data = {}
    global subsample_data
    subsample_data = {}
    global metadata
    metadata = {}

    for video_name in VIDEOS:
        dummy, d = load_video_processed_turker(video_name)
        label_dict[video_name] = d

        with open(os.path.join(OPTIMAL_INPUT_FOLDER, video_name + '_' + str(UNIMPORTANTWORD_THRESH) + '.pickle')) as fh:
            optimal_data[video_name] = pickle.load(fh)

        # baby_lion__tiger_playing_t9je3P-_NmU_greedy_0.8_-1_log.pickle
        with open(os.path.join(SUBSAMPLE_INTPUT_FOLDER, video_name + '_greedy_0.8_-1_log.pickle')) as fh:
            subsample_data[video_name] = pickle.load(fh)

        with open(os.path.join(METADATA_FOLDER, video_name + '.json')) as fh:
            data = json.load(fh)
            metadata[video_name] = data



    for sid, ss in enumerate(start_fid_set):
        # load vis_data
        vis_data = {}
        with open('/home/t-yuche/ranking/gen_rank_data/vis-log/all_vis_' + str(sid) + '.pickle') as fh:
            vis_data = pickle.load(fh)
 
        #for query_str in open('./single_query').readlines():
        for query_str in ['person']:
            query_str = query_str.strip()
            query_ws = query_str.split()
            target_count = len(query_ws)

            metadata_bad_videos = []
            good_videos = []
            good_vis_videos = [] 
            other_videos = []
            videos = list(VIDEOS)

            for w in query_ws:
                if w in blocking_videos.keys():
                    for v in blocking_videos[w]:
                        if v in videos:
                            rindx = videos.index(v)
                            del videos[rindx]
            #videos = list(VIDEOS)
            #random.shuffle(videos)

            video_info = {}
            for vid, video_name in enumerate(videos): 

                rand_fid = ss[video_name]
                start_fid, end_fid = get_video_seg(rand_fid, video_name)
                optimal_key = str(start_fid) + '-' + str(end_fid)

                gt_match_count, gt_non_match_count = has_obj_using_gtlabel(video_name, start_fid, end_fid, query_str)

                optimal_tf = optimal_data[video_name][optimal_key]
                optimal_ws = optimal_tf.keys() 
            
                subsample_tf = subsample_data[video_name][optimal_key]
                subsample_ws = subsample_tf.keys()

                optimal_match_count = 0
                subsample_match_count = 0
                optimal_scores = []
                subsample_scores = []
                for w in query_ws:
                    optimal_scores += [getCosSimilarty({w:1}, optimal_tf)]
                    subsample_scores += [getCosSimilarty({w:1}, subsample_tf)]
                    if w in optimal_ws:
                        optimal_match_count += 1
                    if w in subsample_ws:
                        subsample_match_count += 1 

                optimal_score = np.mean(optimal_scores) 
                subsample_score = np.mean(subsample_scores)

                # get object size
                #print start_fid, end_fid, vis_data[video_name] 
                assert(vis_data[video_name]['start_fid'] == start_fid)
                assert(vis_data[video_name]['end_fid'] == end_fid)

                # video_score
                vis_info = {}
                vis_obj_count = 0
                max_dwell_time = 0
                max_w_h = 0
                caught_it = 0
                for w in query_ws:
                    if w in CLASSES:
                        vis_obj_count += 1 
                        vis_info[w] = {'obj': w, 'ave_obj_size': vis_data[video_name]['vis'][w]['ave_obj_size'], 'dwell_time': vis_data[video_name]['vis'][w]['dwell_time']} 
                        if vis_data[video_name]['vis'][w]['dwell_time']  > max_dwell_time:
                            max_dwell_time = vis_data[video_name]['vis'][w]['dwell_time']
                        ave_size =  vis_data[video_name]['vis'][w]['ave_obj_size']
                        if ave_size[0] > max_w_h:
                            max_w_h = ave_size[0]
                        if ave_size[1] > max_w_h:
                            max_w_h = ave_size[1]

                        if caught_it == 0:
                            for i in vis_data[video_name]['vis'][w]['trace']:
                                if len(i) > 0:
                                    caught_it = 1

                vis_agg_info = {'vis_obj_count' : vis_obj_count, 'max_dwell_time': max_dwell_time, 'max_w_h': max_w_h, 'caught_it': caught_it}

                 
                #if video_name in [x[0] for x in metadata_cand]:
                #    metadata_rank = [x[0] for x in metadata_cand].index(video_name)
                metadata_info = metadata[video_name]
                metadata_score, view_count = compute_metadata_score(metadata_info, query_str)

                cur_videoinfo = {'metadata_score': metadata_score, 'metadata_viewcount': view_count, 'optimal_score': optimal_score, 'subsample_score': subsample_score, 'gt_non_match_count': gt_non_match_count, 'gt_match_count': gt_match_count, 'optimal_match_count': optimal_match_count, 'subsample_match_count': subsample_match_count, 'vis_info': vis_info, 'start_fid': start_fid, 'end_fid': end_fid, 'video_name': video_name, 'vis_agg_info': vis_agg_info}
                video_info[video_name] = cur_videoinfo 
                #print video_name, query_str, start_fid, end_fid
                #print cur_videoinfo
 
            ######################################
            # pick videos
            metadata_bad_videos = []
            metadata_secbad_videos = []
            for video_name in videos:

                if video_info[video_name]['metadata_score'] == 3 and video_info[video_name]['gt_non_match_count'] == target_count and video_info[video_name]['subsample_match_count'] == 0 and video_info[video_name]['optimal_match_count'] == 0:
                    metadata_bad_videos += [video_info[video_name]]

                if query_str == 'sofa':
                    if video_info[video_name]['metadata_score'] ==  0 and video_info[video_name]['gt_non_match_count'] == target_count and video_info[video_name]['subsample_match_count'] == 0 and video_info[video_name]['optimal_match_count'] == 0:
                        metadata_secbad_videos += [video_info[video_name]]
                else:
                    if video_info[video_name]['metadata_score'] > 0 and video_info[video_name]['gt_non_match_count'] == target_count and video_info[video_name]['subsample_match_count'] == 0 and video_info[video_name]['optimal_match_count'] == 0:
                        metadata_secbad_videos += [video_info[video_name]]
                
            metadata_bad_videonames = sorted(metadata_bad_videos, key = lambda x: x['metadata_viewcount'], reverse = True)
            #all_videos = metadata_bad_videonames[: min(2, len(metadata_bad_videonames))]
            all_videos = []
            metadata_secbad_videonames = sorted(metadata_secbad_videos, key = lambda x: x['metadata_viewcount'], reverse = True)
            #for bv in metadata_secbad_videonames:
            #    if len(all_videos) >= 2:
            #        break
            #    all_videos += [bv]
            print 'Has', len(all_videos), 'bad videos'
            max_video_count = 0
            for v in all_videos:
                if v['metadata_viewcount'] > max_video_count:
                    max_video_count = v['metadata_viewcount']

            #######################################

            good_vis_videos = []
            other_videos = []
            rest_videos = []
            tolerate_videos = []
            subsample_better_videos = []
            last_resort_videos = []
            for video_name in videos:
                if video_info[video_name]['gt_match_count'] == target_count and video_info[video_name]['subsample_match_count'] == target_count and video_info[video_name]['optimal_match_count'] == target_count and (video_info[video_name]['vis_agg_info']['max_dwell_time'] > 0 or video_info[video_name]['vis_agg_info']['caught_it']):

                    good_vis_videos += [video_info[video_name]]
                

                #elif video_info[video_name]['gt_match_count'] == target_count and video_info[video_name]['subsample_match_count'] == target_count and video_info[video_name]['optimal_match_count'] == target_count:
                #    other_videos += [video_info[video_name]]
               
                #elif video_info[video_name]['gt_match_count'] > 0 and video_info[video_name]['subsample_match_count'] == target_count and video_info[video_name]['optimal_match_count'] == target_count:
                #    rest_videos += [video_info[video_name]]
                #elif video_info[video_name]['gt_match_count'] > 0 and video_info[video_name]['subsample_match_count'] > 0 and video_info[video_name]['optimal_match_count'] > 0:
                    
                #    tolerate_videos += [video_info[video_name]]
                elif video_info[video_name]['gt_match_count'] > 0 and video_info[video_name]['subsample_match_count'] > 0 and (video_info[video_name]['vis_agg_info']['max_dwell_time'] > 0 or video_info[video_name]['vis_agg_info']['caught_it']):
                    subsample_better_videos += [video_info[video_name]] 
                elif video_info[video_name]['gt_match_count'] > 0 and video_info[video_name]['optimal_match_count'] > 0 and (video_info[video_name]['vis_agg_info']['max_dwell_time'] > 0 or video_info[video_name]['vis_agg_info']['caught_it']):
                    last_resort_videos += [video_info[video_name]] 

            print 'good vis videos:', len(good_vis_videos)
            #print 'other videos:', len(other_videos)
            print 'subsample better videos', len(subsample_better_videos)
            print 'last resort videos:', len(last_resort_videos)
            good_vis_videonames = sorted(good_vis_videos, key = lambda x: x['vis_agg_info']['vis_obj_count'], reverse = True) 
            all_videos += good_vis_videonames[:min(3, len(good_vis_videonames))]
           

            good_vis_videonames = sorted(good_vis_videos, key = lambda x: x['vis_agg_info']['max_dwell_time'], reverse = True) 
            counter = 0
            for v in good_vis_videonames:
                if v['video_name'] not in [x['video_name'] for x in all_videos]:
                    if counter == 3:
                        break
                    all_videos += [v]
                    counter += 1

            good_vis_videonames = sorted(good_vis_videos, key = lambda x: x['vis_agg_info']['max_w_h'], reverse = True)
            counter = 0
            for v in good_vis_videonames:
                if v['video_name'] not in [x['video_name'] for x in all_videos]:
                    if counter == 2:
                        break
                    all_videos += [v]
                    counter += 1

            good_vis_videonames = sorted(good_vis_videos, key = lambda x: x['vis_agg_info']['max_w_h'])
            counter = 0
            for v in good_vis_videonames:
                if v['video_name'] not in [x['video_name'] for x in all_videos]:
                    if counter == 2:
                        break
                    all_videos += [v]
                    counter += 1
            
            print ' ------ good vis videos:', len(all_videos)
            ## if multiple objects
            #for video_name in VIDEOS: 

            other_videonames = sorted(other_videos, key = lambda x: x['metadata_viewcount'])

            for v in other_videonames: 
                if len(all_videos) >= 10:
                    break
                if v['video_name'] not in [x['video_name'] for x in all_videos]:
                    all_videos += [v]
            
            print ' ------ better metadata acc:', len(all_videos)
     
            rest_videonames = sorted(rest_videos, key = lambda x: x['gt_match_count'], reverse = True) 
            for v in rest_videonames: 
                if len(all_videos) >= 10:
                    break
                if v['video_name'] not in [x['video_name'] for x in all_videos]:
                    all_videos += [v]
     
            tolerate_videonames = sorted(tolerate_videos, key = lambda x: x['gt_match_count'], reverse = True) 
            for v in rest_videonames: 
                if len(all_videos) >= 10:
                    break
                if v['video_name'] not in [x['video_name'] for x in all_videos]:
                    all_videos += [v]
     
            subsample_better_videonames = sorted(subsample_better_videos, key = lambda x: x['gt_match_count'], reverse = True) 
            for v in subsample_better_videonames: 
                if len(all_videos) >= 10:
                    break
                if v['video_name'] not in [x['video_name'] for x in all_videos]:
                    all_videos += [v]
            
            if query_str == 'sofa':
                for v in good_vis_videonames:
                    if len(all_videos) >= 10:     
                        break
                    if v['video_name'] not in [x['video_name'] for x in all_videos]:
                        all_videos += [v]

            last_resort_videonames = sorted(last_resort_videos, key = lambda x: x['metadata_viewcount']) 
            for v in last_resort_videonames:
                    if len(all_videos) >= 10:
                        break 
                    if v['video_name'] not in [x['video_name'] for x in all_videos]:
                        all_videos += [v]           
 
            print query_str, sid, len(all_videos) 

            with open(os.path.join(OUTPUT_FOLDER, query_str + '_' + str(sid) + '.pickle'), 'wb') as fh:
                pickle.dump(all_videos[:10], fh)
            #print sid, query_str, all_videos
