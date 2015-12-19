import os
import pickle
import math
import numpy as np
import random
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def get_center(bbox):
    return ((bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2)
    
def get_dist(x, y, w, h):
    return math.sqrt( ((x[0] - y[0])/(w * 1.0))** 2 + ((x[1]-y[1])/(h * 1.0)) ** 2)


def get_center_dist(a, b, w, h):
    if a == None or b == None:
        return -1
    a_c = get_center(a)
    b_c = get_center(b)
    
    return get_dist(a_c, b_c, w,h)


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

def get_gt_ranking(gt_ranking, query_infos):
    gt_ranking_w_score = {}
    for ridx, r in enumerate(gt_ranking):
        pos = sorted([x for x, letter in enumerate(r) if letter == '_'])
        video_name = r[:pos[-2]] 
        gt_ranking_w_score[video_name] = 10 - ridx

    return gt_ranking_w_score

def get_e2e_speed(trace, w, h):
    first_bbx = None
    last_bbx = None
    for i in trace:
        if len(i) > 0 and first_bbx == None:
            first_bbx = get_biggest_obj(i)
        if len(i) > 0:
            last_bbx = get_biggest_obj(i)

    return get_center_dist(last_bbx, first_bbx, w, h)

def get_obj_num(trace):

    obj_num_log = []    
    for i in trace:
        if len(i) > 0:
            obj_num_log += [len(i)]

    if len(obj_num_log) == 0:
        return 0 
    return np.mean(obj_num_log)

def compose_train_example(label_score, vis_data, query_str, query_info, fps_w_h , scheme = 'sub'):

    fps = fps_w_h[0]
    w = fps_w_h[1]
    h = fps_w_h[2]

    queries = query_str.split()
    dwell_times = []
    ave_speeds = []
    e2e_speeds =[]
    obj_ave_sizes = []
    obj_max_sizes = []
    num_objs = []
    obj_max_w_hs = []
    for query in queries:
        if query in CLASSES:
            #print vis_data['vis'][query]
            dwell_time = vis_data['vis'][query]['dwell_time']/150.
            moving_speed = vis_data['vis'][query]['moving_speed']
            e2e_speed =  get_e2e_speed(vis_data['vis'][query]['trace'], w, h) 
            obj_ave_size = vis_data['vis'][query]['ave_obj_size'][0] * vis_data['vis'][query]['ave_obj_size'][1]
            obj_max_size = vis_data['vis'][query]['max_obj_size'][0] * vis_data['vis'][query]['max_obj_size'][1]
            num_obj = get_obj_num(vis_data['vis'][query]['trace']) 
            obj_max_w_h = query_info['vis_agg_info']['max_w_h']
             
            dwell_times += [dwell_time] 
            ave_speeds += [moving_speed]
            e2e_speeds += [e2e_speed]
            obj_ave_sizes += [obj_ave_size]
            obj_max_sizes += [obj_max_size]
            num_objs += [num_obj]
            obj_max_w_hs += [obj_max_w_h]
    
   # print dwell_times, ave_speeds, e2e_speeds, obj_ave_sizes, obj_max_sizes, num_objs 
    if scheme == 'sub':
        text_score = query_info['subsample_score'] 
    else:
        text_score = query_info['optimal_score'] 

       
    train_data =  {'score': label_score, 'dwell_time': np.mean(dwell_times), 'ave_speed': np.mean(ave_speeds), 'e2e_speed': np.mean(e2e_speeds), 'obj_ave_size': np.mean(obj_ave_sizes), 'obj_max_size': np.mean(obj_max_sizes), 'obj_num': np.mean(num_objs), 'obj_max_w_h': np.mean(obj_max_w_hs)}  
    #print train_data
    #print ''
     
    return train_data 

FEATURES = ['dwell_time', 'ave_speed', 'e2e_speed', 'obj_ave_size', 'obj_max_size', 'obj_num', 'obj_max_w_h']  
if __name__ == "__main__":

    global stream_rates
    global start_fid_set
    global query_video_list
    global optimal_rank

    INPUT_FOLDER = '/home/t-yuche/ranking/gen_rank_data/label_videos'
    query_video_list = {}
    
    for f in os.listdir(INPUT_FOLDER):
        #print f
        segs = f.split('.')[0].split('_')
        query = segs[0]
        sid = segs[1]
        
        with open(os.path.join(INPUT_FOLDER, f)) as fh:
            cand_videos = pickle.load(fh)
            query_video_list[query + '-' + str(sid)] = cand_videos

    OPTIMAL_RANK_FOLDER = '/home/t-yuche/ranking/eval/opt-rank' 
    optimal_rank = {}
    for f in os.listdir(OPTIMAL_RANK_FOLDER):
        segs = f.split('.')[0].split('_')
        query = segs[0]
        sid = segs[1]
        with open(os.path.join(OPTIMAL_RANK_FOLDER, f)) as fh:
            optimal_rank[query + '-' + str(sid)] = pickle.load(fh)

    rewrite_data = {}
    with open('rewrite_vision_info')as fh:
        lines = fh.readlines()
        for line in lines:
            line = line.strip()
            segs = line.split()
            video_name = segs[0]
            original_label = segs[1]
            tolabel = segs[2]
            rewrite_data[video_name] = (original_label, tolabel)

    video_fps_wh_info = {}
    with open('video_metadata.pickle') as fh:
        video_fps_wh_info = pickle.load(fh)
    ### done with loading ###  

    ## extracting features         
    labeled_queries = optimal_rank.keys()
    #scores = {METADATA: [], OPT_TXT_VIS: [], OPT_TXT: [], SUB_TXT_VIS: [], SUB_TXT: []}
    
    # separate training/testing data
    TRAINING_PORTION = 1
    #random.seed(0)
    #train_queries = random.sample(labeled_queries, int(len(labeled_queries) * 0.5))
    train_queries = [q for q in labeled_queries if (int(q.split('-')[1]) >= 0 and int(q.split('-')[1]) <= 4) ]
    test_queries = [] 
    for label_query in labeled_queries:
        if label_query not in train_queries:
            test_queries += [label_query]
    print len(train_queries), len(test_queries)

    ## compose train_data
    training_data = []
    
    trainoutpath = 'train_data'
    train_fh = open(trainoutpath, 'w')
    for label_query in train_queries:
        print label_query
        query = label_query.split('-')[0]
        ssid = label_query.split('-')[1]

        vis_data = {}
        with open('/home/t-yuche/ranking/gen_rank_data/vis-log/all_vis_' + str(ssid) + '.pickle') as fh:
            vis_data = pickle.load(fh)


        for video_name in rewrite_data.keys():
            a_label = rewrite_data[video_name][0]
            b_label = rewrite_data[video_name][1]

            tmp_data = dict(vis_data[video_name]['vis'][b_label])
            vis_data[video_name]['vis'][b_label] = vis_data[video_name]['vis'][a_label]
            vis_data[video_name]['vis'][a_label] = tmp_data

        query_infos = query_video_list[label_query]
        optimal_ranking = optimal_rank[label_query]
        video_scores = get_gt_ranking(optimal_ranking, query_infos)

        for video_name in video_scores:
            query_info = [x for x in query_infos if x['video_name'].find(video_name) >= 0][0]
            data = compose_train_example(video_scores[video_name], vis_data[video_name], query, query_info, video_fps_wh_info[video_name])
            training_data += [data]
            
            outstr = ','.join([str(data[feature]) for feature in FEATURES])
            train_fh.write(str(CLASSES.index(query.split()[0])) + ',' + outstr + ',' + str(data['score']) + '\n')

        #exit()
    train_fh.close()
    ## compose test_data
    '''
    test_data = []
    print 'Composing testing data..'
    for label_query in test_queries:
        print label_query
        query = label_query.split('-')[0]
        ssid = label_query.split('-')[1]

        vis_data = {}
        with open('/home/t-yuche/ranking/gen_rank_data/vis-log/all_vis_' + str(sid) + '.pickle') as fh:
            vis_data = pickle.load(fh)


        for video_name in rewrite_data.keys():
            a_label = rewrite_data[video_name][0]
            b_label = rewrite_data[video_name][1]

            tmp_data = dict(vis_data[video_name]['vis'][b_label])
            vis_data[video_name]['vis'][b_label] = vis_data[video_name]['vis'][a_label]
            vis_data[video_name]['vis'][a_label] = tmp_data

        query_infos = query_video_list[label_query]
        optimal_ranking = optimal_rank[label_query]
        video_scores = get_gt_ranking(optimal_ranking, query_infos)

        for video_name in video_scores:
            query_info = [x for x in query_infos if x['video_name'].find(video_name) >= 0][0]
            data = compose_train_example(video_scores[video_name], vis_data[video_name], query, query_info, video_fps_wh_info[video_name])
            test_data += [data]
     

    # train
    X = []
    Y = []
    for data_dict in train_data:
        point = [data_dict[feature] for feature in FEATURES]
        X += [point]

    # test
    
    regr = DecisionTreeRegressor(max_depth = 2)
    regr.fit(X, Y)


    ## testing
    #for data_dict in test_queries: 
    '''
