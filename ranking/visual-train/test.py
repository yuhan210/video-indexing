import os
import pickle
import math
import numpy as np
import random
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from ndcg import *
import matplotlib
import operator
import matplotlib.pyplot as plt
from utils import *

font = {'family' : 'normal',
        'size'   : 14,
    }
text = {'color': 'black'}
line = {'linewidth': 3}
matplotlib.rc('font', **font)
matplotlib.rc('text', **text)
matplotlib.rc('lines', **line)


METADATA = 'Metadata'
OPT_TXT = 'Text Only \n (No Subsampling)'
OPT_TXT_VIS = 'Text + \n Visual-Hint \n (No Subsampling)'
SUB_TXT = 'Text Only \n (Subsampling)'
SUB_TXT_VIS = 'Panorama'

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
        gt_ranking_w_score[video_name] = len(gt_ranking) - ridx

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

    #print query_info
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
     

def optimal_text(stream_infos):
    sorted_opt_text = sorted(stream_infos, key = lambda x: x['optimal_score'], reverse = True)
 
    return [(x['video_name'], x['optimal_score']) for x in sorted_opt_text]
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


def get_opt_text_score(query, video_name, start_fid, end_fid, optimal_data):
    #print video_name, optimal_data[video_name].keys()
    optimal_key = start_fid + '-' + end_fid
    optimal_tf = optimal_data[video_name][optimal_key]
    optimal_ws = optimal_tf.keys()
    query_ws = query.split()

    optimal_scores = []
    for w in query_ws:
        optimal_scores += [getCosSimilarty({w:1}, optimal_tf)]

    optimal_score = np.mean(optimal_scores) 
    return optimal_score

def subsample_text(stream_infos):
    sorted_opt_text = sorted(stream_infos, key = lambda x: x['subsample_score'], reverse = True)
 
    return [(x['video_name'] , x['subsample_score']) for x in sorted_opt_text]

def is_null_sample(point):
    if len([x for x in point if x < 0]) > 0:
        return True

    # [14, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0] 
    if len([x for x in point if x == 0]) > 4:
        return True
    return False

FEATURES = ['dwell_time', 'ave_speed', 'e2e_speed', 'obj_ave_size', 'obj_max_size', 'obj_num', 'obj_max_w_h']  
if __name__ == "__main__":

    score_log = []
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

    
    with open('/home/t-yuche/ranking/visual-train/models/linear_regr_3.pickle') as fh:
        model = pickle.load(fh)

    OPTIMAL_INPUT_FOLDER = '/home/t-yuche/admission-control/eval/optimal-log/optimal-tf'
    optimal_data = {}
    for video in open('/home/t-yuche/video_list.txt').read().split():

        with open(os.path.join(OPTIMAL_INPUT_FOLDER, video + '_0.5.pickle')) as fh:
            optimal_data[video] = pickle.load(fh)
            
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
    #print len(train_queries), len(test_queries)

    ## compose train_data
    '''
    training_data = []
    
    trainoutpath = 'train_data'
    train_fh = open(trainoutpath, 'w')
    for label_query in train_queries:
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
            training_data += [data]
            
            outstr = ','.join([str(data[feature]) for feature in FEATURES])
            train_fh.write(outstr + ',' + str(data['score']) + '\n')

    train_fh.close()
    '''
    ## compose test_data
    test_data = []
    scores = {METADATA: [], OPT_TXT_VIS: [], OPT_TXT: [], SUB_TXT_VIS: [], SUB_TXT: []}
    print 'Composing testing data..'
    for label_query in test_queries:
        print '-----------------------'
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
        #print query_infos
        gt_ranking = optimal_rank[label_query]
        no_start_end_fid_gt_ranking = []

        opt_text_scores = {}
        for r in gt_ranking:
            cur_video_name, start_fid, end_fid = remove_start_end_fid(r)   
            no_start_end_fid_gt_ranking += [cur_video_name]      
            opt_text_scores[cur_video_name] = get_opt_text_score(query, cur_video_name, start_fid, end_fid, optimal_data) 
 
        tmp = sorted(opt_text_scores.items(), key = operator.itemgetter(1), reverse = True)
        opt_text_ranking = [x[0] for x in tmp]
        #opt_text_ranking = [x[0] for x in optimal_text(query_infos)]
        sub_text_ranking = [x[0] for x in subsample_text(query_infos)]
        video_scores = get_gt_ranking(gt_ranking, query_infos)

       
        video_pred_scores = {}
        opt_video_pred_scores = {} 
        for video_name in video_scores:
            query_info = [x for x in query_infos if x['video_name'].find(video_name) >= 0][0]

            data_dict = compose_train_example(video_scores[video_name], vis_data[video_name], query, query_info, video_fps_wh_info[video_name])
            opt_data_dict = compose_train_example(video_scores[video_name], vis_data[video_name], query, query_info, video_fps_wh_info[video_name], 'opt')

            point = [CLASSES.index(query.split()[0])] + [data_dict[feature] for feature in FEATURES]
            opt_point = [CLASSES.index(query.split()[0])] + [opt_data_dict[feature] for feature in FEATURES]
           
            if is_null_sample(point):
                score = -1
            else: 
                score = model.predict(point)[0]

            if is_null_sample(point):
                opt_score = -1
            else:
                opt_score = model.predict(opt_point)[0]
            #print video_name, point, score
            video_pred_scores[video_name] =  score
            opt_video_pred_scores[video_name] = opt_score

        #print opt_pred_video_score
        opt_pred_video_score = sorted(opt_video_pred_scores.items(), key = operator.itemgetter(1), reverse = True)
        opt_pred_video_rank = [x[0] for x in opt_pred_video_score if x[1] > 0]
       
        pred_video_score = sorted(video_pred_scores.items(), key = operator.itemgetter(1), reverse = True)
        pred_video_rank = [x[0] for x in pred_video_score if x[1] > 0]
        
        #print pred_video_rank
        intersect_video = list(set(no_start_end_fid_gt_ranking) & set(pred_video_rank))
        tmp_vs = list(no_start_end_fid_gt_ranking)
        tt = list(no_start_end_fid_gt_ranking)
        for v in tmp_vs:
            if v not in intersect_video:
                tt.remove(v)
        print 'gt:' 
        print tt
        if len(tt) <= 1:
            continue
        '''
        sub_text_vis_rank_score = get_Spearmansfr_score(no_start_end_fid_gt_ranking, pred_video_rank)
        sub_text_rank_score = get_Spearmansfr_score(no_start_end_fid_gt_ranking, sub_text_ranking)
        opt_text_vis_rank_score = get_Spearmansfr_score(no_start_end_fid_gt_ranking, opt_pred_video_rank)
        opt_text_rank_score = get_Spearmansfr_score(no_start_end_fid_gt_ranking, opt_text_ranking)
        '''
        sub_text_vis_rank_score = get_Spearmansfr_intersect_score(no_start_end_fid_gt_ranking, pred_video_rank, intersect_video, 'sub text vis')
        sub_text_rank_score = get_Spearmansfr_intersect_score(no_start_end_fid_gt_ranking, sub_text_ranking, intersect_video, 'sub text')
        opt_text_vis_rank_score = get_Spearmansfr_intersect_score(no_start_end_fid_gt_ranking, opt_pred_video_rank, intersect_video, 'opt text vis')
        opt_text_rank_score = get_Spearmansfr_intersect_score(no_start_end_fid_gt_ranking, opt_text_ranking, intersect_video, 'opt text')
        print 'sub_text_vis', sub_text_vis_rank_score
        print 'sub_text', sub_text_rank_score
        print 'opt_text_vis', opt_text_vis_rank_score
        print 'opt_text', opt_text_rank_score

        scores[OPT_TXT] += [opt_text_rank_score]
        scores[OPT_TXT_VIS] += [opt_text_vis_rank_score]
        scores[SUB_TXT] += [sub_text_rank_score]
        scores[SUB_TXT_VIS] += [sub_text_vis_rank_score]

        score_log += [{label_query: [opt_text_vis_rank_score, opt_text_rank_score, sub_text_vis_rank_score, sub_text_rank_score]}]
    fig = plt.figure()
    x_labels = [METADATA, OPT_TXT, OPT_TXT_VIS, SUB_TXT, SUB_TXT_VIS]
    ax = fig.add_subplot(111)
    width = 0.7
    ind = [x  for x in range(len(scores))]
    ax.bar( ind , [np.mean(scores[x]) * 100 for x in x_labels], width, yerr= [np.std(scores[x])*100 for x in x_labels])
    ax.set_xticks( [x + width/2 for x in ind] )
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Ranking Accuracy (%)')
    fig.savefig('ranking-results.pdf', bbox_inches = 'tight')
    with open('score_log.pickle','wb') as  fh:
        pickle.dump(score_log, fh)
