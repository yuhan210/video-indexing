import os
import csv
import itertools
import pickle
from utils import *
import random
import operator
import matplotlib
import matplotlib.pyplot as plt
try:
    plt.style.use('ggplot')
except:
    pass
font = {
        'size'   : 18,
    }
text = {'color': 'black'}
line = {'linewidth': 3}
matplotlib.rc('font', **font)
matplotlib.rc('text', **text)

all_bad_videos = open('dog_bad').read().split()
def load_turker_video_labels(file_name):
    csv_file = open(file_name)
    csv_reader = csv.DictReader(csv_file, delimiter="\t") 
   
    selection_info = []
    for row in csv_reader:
        if 'Answer.query_str' not in row:
            query_str = 'dog'
        else:
            query_str = row['Answer.query_str']
        video_name_a = row['Answer.video_name_a']
        video_name_b = row['Answer.video_name_b']
        selection = row['Answer.selected_value'] 

        if len(selection) == 0:
            continue

        key = video_name_a + '_' + video_name_b
        selection_info += [{'va': video_name_a, 'vb':video_name_b, 'selection': selection, 'query_str':query_str}]


    return selection_info
def stem_x(edges):

    bin_num = len(edges) -1
    x = []
    for i in xrange(len(edges)):
        if i == len(edges) - 1:
            break
    
        x+= [(edges[i]  + edges[i+1])/(2 * 1.0)]
        

    return x


def get_bin_idx(edges, value):
    for i in xrange(len(edges) -1):
        if value >= edges[i] and value < edges[i+1]:
            return i

    if value == edges[-1]:
        return len(edges) -2

def plot_fig(sel_pairs, query, x_label,V = 1,NUMBER_BINS =4):
   
    #sel_pairs = random.sample(sel_pairs, 1131) 
    fig = plt.figure()
    xx = [x[0] for x in sel_pairs]
    xx_hist, xx_edges = np.histogram(xx, bins=NUMBER_BINS) 
    xx_stem_x = stem_x(xx_edges)

    xx_count = [0] * NUMBER_BINS
    for p in sel_pairs:
        if p[1] == V:
            idx = get_bin_idx(xx_edges, p[0])
            xx_count[idx] += 1
    
    print xx_count
    print xx_hist, sum(xx_hist)
    print xx_edges, xx_stem_x
    width = xx_stem_x[1] - xx_stem_x[0]
    for idx, c in enumerate(xx_count):
        xx_count[idx] /= (xx_hist[idx]  * 1.0)
    for idx, pos in enumerate(xx_stem_x):
        plt.text(pos - width/3., (xx_count[idx] + 0.015) * 100, '(' + str(int(xx_count[idx] * xx_hist[idx])) + '/' + str(xx_hist[idx]) + ')', fontsize = 15)
    width = xx_stem_x[1] - xx_stem_x[0]
    ax = fig.add_subplot(111)
    #ax.stem(xx_stem_x, [100 * x for x in xx_count], '-.')
    ax.bar(xx_stem_x, [100 * x for x in xx_count], align = 'center', width = xx_stem_x[1] - xx_stem_x[0])
    ax.axhline(y=50,xmin=0,xmax=100,c="red",linewidth=2,zorder=1, ls='dotted')
    ax.set_ylabel('Probability of being Selecting as "Relevant" (%)')
    ax.set_xlim([xx_stem_x[0] - width/2., xx_stem_x[-1] + width/2.]) 
    ax.set_xlabel(x_label)
    ax.set_ylim([0, 100])
    ax.set_title('keyword:' + query, fontsize = 18)
    plt.savefig('./figs/' + query + '_' + x_label + '.pdf', bbox_inches = 'tight')
    plt.show()
def count(selections):

    video_names = list(set([x['va'] for x in selections] + [x['vb'] for x in selections]))
    select_count = {}
    for x in video_names:
        select_count[x] = 0
    for video_name in video_names:
        for selection in selections:
            if selection['va'] == video_name and selection['selection'] == 'A':
                select_count[video_name] += 1
          
            if selection['vb'] == video_name and selection['selection'] == 'B':
                select_count[video_name] += 1
        #print video_name, select_count[video_name] 

    #print select_count

def get_rank_score(rank, selections):

    satisfy_count = 0
    violate_videos = []
    for selection in selections:
        if selection['va'] in rank and selection['vb'] in rank:
            idx_a = rank.index(selection['va'])
            idx_b = rank.index(selection['vb'])

            if selection['selection'] == 'A': 
                if idx_a < idx_b:
                    satisfy_count += 1
                else:
                    if selection['va'] not in violate_videos:
                        violate_videos += [selection['va']]
                    if selection['vb'] not in violate_videos:
                        violate_videos += [selection['vb']]
            else:
                if idx_a > idx_b:
                    satisfy_count += 1
                else:
                    if selection['va'] not in violate_videos:
                        violate_videos += [selection['va']]
                    if selection['vb'] not in violate_videos:
                        violate_videos += [selection['vb']]

    #print '-----', rank, satisfy_count
    return satisfy_count, violate_videos

'''
def get_rank_score(rank, selections):

    violate_count = len(selections)
    for selection in selections:
        if selection['va'] in rank and selection['vb'] in rank:
            idx_a = rank.index(selection['va'])
            idx_b = rank.index(selection['vb'])

            if selection['selection'] == 'A': 
                if idx_a > idx_b:
                    violate_count += 1
            else:
                if idx_a < idx_b:
                    violate_count += 1
    return violate_count
'''
def get_best_position(rank, video_name, selections):
    _rank = list(rank)
    max_score = 0
    max_position = -1
    for i in xrange(len(rank) + 1):
        tmp_rank = list(_rank)
        tmp_rank.insert(i, video_name)
        score, dummy = get_rank_score(tmp_rank, selections)
        
        if score > max_score:
            max_position = i
            max_score = score

    return max_score, max_position

def get_betterone(video_a, video_b, selections):
    for selection in selections:
        if (selection['va'] == video_a and selection['vb'] == video_b):
            if selection['selection'] == 'A':
                return 0
            else:
                return 1
        elif (selection['va'] == video_b and selection['vb'] == video_a):
            if selection['selection'] == 'A':
                return 1
            else:
                return 0

def is_bad_video(bad_videos, v):

    for bv in bad_videos:
        if v.find(bv) >= 0:
            return True

    return False

def show_obj_size(video_info, voted_info, query):
    bad_videos = []
    #bad_videos = open('os_badv').read().split()
    bad_videos += all_bad_videos

    os_sel_pairs = []
    bad_v_count = {}
    for pair_comp in voted_info:

        va = pair_comp['va']
        vb = pair_comp['vb']
        sel = pair_comp['selection']  
        va_key = va + '-' + query
        vb_key = vb + '-' + query

        if va_key not in video_info.keys() or vb_key not in video_info.keys() or is_bad_video(bad_videos, va) or is_bad_video(bad_videos, vb):
            continue

        va_os = video_info[va_key]['obj_size'] 
        vb_os = video_info[vb_key]['obj_size']
    
        va_dt = video_info[va_key]['dwell_time'] 
        vb_dt = video_info[vb_key]['dwell_time']
        
        va_md = video_info[va_key]['moving_dist'] 
        vb_md = video_info[vb_key]['moving_dist']
     
    
        value = abs(va_os - vb_os)     
        if abs(va_dt - vb_dt) < 150 and abs(va_md - vb_md) < 1:

            if va_os > vb_os:
                if sel == 'A':
                    os_sel_pairs += [(va_os - vb_os, 1)] 
                else:
                    if value > 0.7 and value < 0.9:
                        if va in bad_v_count.keys():
                            bad_v_count[va]  += 1
                        else:
                            bad_v_count[va] = 1
                    os_sel_pairs += [(va_os - vb_os, 0)] 
            else:
                if sel == 'B':
                    os_sel_pairs += [(vb_os - va_os, 1)] 
                else:    
                    if value > 0.7 and value < 0.9:
                        print vb, va, value
                        if vb in bad_v_count.keys():
                            bad_v_count[vb]  += 1
                        else:
                            bad_v_count[vb] = 1
                    os_sel_pairs += [(vb_os - va_os, 0)] 
    print len(os_sel_pairs)
    os_sel_pairs += dog_os_sel_pairs
    sorted_x = sorted(bad_v_count.items(), key=operator.itemgetter(1), reverse =True)
    print sorted_x[:10]
    plot_fig(os_sel_pairs, query,'Normalized Object Size Difference')

def show_moving_dist(video_info, voted_info, query):

    
    bad_videos = []
    #bad_videos = open('md_badv').read().split()
    bad_videos += all_bad_videos
    bad_v_count = {}
    md_sel_pairs = []
    for pair_comp in voted_info:

        va = pair_comp['va']
        vb = pair_comp['vb']
        sel = pair_comp['selection']  
        va_key = va + '-' + query
        vb_key = vb + '-' + query

        if va_key not in video_info.keys() or vb_key not in video_info.keys() or is_bad_video(bad_videos, va) or is_bad_video(bad_videos, vb):
            continue

        va_os = video_info[va_key]['obj_size'] 
        vb_os = video_info[vb_key]['obj_size']
    
        va_dt = video_info[va_key]['dwell_time'] 
        vb_dt = video_info[vb_key]['dwell_time']
        
        va_md = video_info[va_key]['moving_dist'] 
        vb_md = video_info[vb_key]['moving_dist']
     
    
        value = abs(va_md - vb_md)     
        if abs(va_os - vb_os) < 1 and abs(va_dt - vb_dt) < 150:

            if va_md < vb_md:
                if sel == 'A':
                    md_sel_pairs += [(va_md - vb_md, 1)] 
                else:
                    if value > 0.45  and value < 1:
                        if va in bad_v_count.keys():
                            bad_v_count[va] += 1
                        else:
                            bad_v_count[va] = 1
                    md_sel_pairs += [(va_md - vb_md, 0)] 
            else:
                if sel == 'B':
                    md_sel_pairs += [(vb_md - va_md, 1)] 
                else: 
                    if value > 0.45 and value < 1:
                        if vb in bad_v_count.keys():
                            bad_v_count[vb] += 1
                        else:
                            bad_v_count[vb] = 1
                    md_sel_pairs += [(vb_md - va_md, 0)] 
    sorted_x = sorted(bad_v_count.items(), key=operator.itemgetter(1), reverse =True)
    print sorted_x[:10]
    md_sel_pairs += dog_md_sel_pairs    

    #print os_sel_pairs
    plot_fig(md_sel_pairs, query,'Normalized Moving Distance Difference', 1)

def show_dwell_time(video_info, voted_info, query):

    #bad_videos = open('dt_badv').read().split()
    bad_videos = []
    bad_videos += all_bad_videos
    #bad_videos = []
    bad_v_count = {}
    dt_sel_pairs = []
    for pair_comp in voted_info:

        va = pair_comp['va']
        vb = pair_comp['vb']
        sel = pair_comp['selection']  
        va_key = va + '-' + query
        vb_key = vb + '-' + query

        if va_key not in video_info.keys() or vb_key not in video_info.keys() or is_bad_video(bad_videos, va) or is_bad_video(bad_videos, vb):
            continue

        va_os = video_info[va_key]['obj_size'] 
        vb_os = video_info[vb_key]['obj_size']
    
        va_dt = video_info[va_key]['dwell_time'] 
        vb_dt = video_info[vb_key]['dwell_time']
        
        va_md = video_info[va_key]['moving_dist'] 
        vb_md = video_info[vb_key]['moving_dist']
     
    
        value = abs(va_dt - vb_dt)/150.
        if abs(va_os - vb_os) < 100 and abs(va_md - vb_md) < 100:

            if va_dt > vb_dt:
                if sel == 'A':
                    dt_sel_pairs += [((va_dt - vb_dt)/150., 1)] 
                else:
                    if value > 0.7 and value < 1:
                        if va in bad_v_count.keys():
                            bad_v_count[va]+= 1
                        else:
                            bad_v_count[va] = 1
                    dt_sel_pairs += [((va_dt - vb_dt)/150., 0)] 
            else:
                if sel == 'B':
                    dt_sel_pairs += [((vb_dt - va_dt)/150., 1)] 
                else: 
                    if value >0.7 and value < 1:
                        if vb in bad_v_count.keys():
                            bad_v_count[vb]+= 1
                        else:
                            bad_v_count[vb] = 1
                    dt_sel_pairs += [((vb_dt - va_dt)/150., 0)] 

    sorted_x = sorted(bad_v_count.items(), key=operator.itemgetter(1), reverse =True)
    #print sorted_x[:10]
    print '---------'
    for i in sorted_x:
        if i[1] < 1:
            break
        print i[0]
    print '--------'
    dt_sel_pairs += dog_dt_sel_pairs
    #print os_sel_pairs
    plot_fig(dt_sel_pairs, query, 'Normalized Dwell Time Difference')

def rank(selections):

    video_names = list(set([x['va'] for x in selections] + [x['vb'] for x in selections]))
    rank = [video_names[0]] 

    while len(rank) < len(video_names):
        max_score = -1
        max_position = [-1, -1]
        for video_name in video_names: # pick the best video
            if video_name not in rank:  
                score, position = get_best_position(rank, video_name, selections) 
                #print video_name, position, score
                if score > max_score:
                    max_score = score
                    max_position = [video_name, position]
                elif score == max_score and position == max_position[1]:
                    
                    b_idx = get_betterone(max_position[0], video_name, selections)
                    if b_idx == 1:
                        max_score = score
                        max_position = [video_name, position]
 

        #print 'inserting', max_position[0], 'in position', max_position[1]
        rank.insert(max_position[1], max_position[0])

    return rank

def swapping(rank, selections):
    print 'SWAPPING'

    score, bad_vs = get_rank_score(rank, selections)
    indexes = [rank.index(x) for x in bad_vs]
    min_idx = min(indexes) - 1
    max_idx = max(indexes) + 1
    #print max_idx - min_idx + 1
     
    for p in itertools.permutations(range(min_idx, max_idx + 1)):
    
        tmp_rank = list(rank)
        for c, i in enumerate(xrange(min_idx, max_idx+1)):
            tmp_rank[i] = rank[p[c]]
        score, dummy = get_rank_score(tmp_rank, selections)
        if score == 45:
            return tmp_rank
            break
    

def bruteforce(selections):
    video_names = list(set([x['va'] for x in selections] + [x['vb'] for x in selections]))

    n_videos = len(video_names)
    for p in itertools.permutations(range(n_videos)):
        rank = [video_names[x] for x in p]  
        score, dummy = get_rank_score(rank, selections)
        if score == 45:
            break 
        print p, score

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

def check_video_chunk(chunk, w, h, query):

    #OBJ_NAME = ['dog', 'horse', 'cat', 'bird']
    OBJ_NAME = [query]
    chunk_info = {}

    counts = get_obj_count(chunk, OBJ_NAME)
    chunk_info['dwell_time'] = len([x for x in counts if x > 0])
    chunk_info['obj_count']  = 1

    first_obj = None
    last_obj = None
    obj_sizes = []
    for idx, item in enumerate(chunk):
        # for each frame
        for obj in item['pred']: 
            # for each prediction 
            if obj['class'] in OBJ_NAME and obj['score'] > 0.5:
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

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


if __name__ == "__main__":

    global dog_os_sel_pairs
    global dog_dt_sel_pairs
    global dog_md_sel_pairs

    with open('../analysis/os.pickle') as fh:
        dog_os_sel_pairs = pickle.load(fh)
    with open('../analysis/dt.pickle') as fh:
        dog_dt_sel_pairs = pickle.load(fh)
    with open('../analysis/md.pickle') as fh:
        dog_md_sel_pairs = pickle.load(fh)


    LABEL_FOLDER = './turker_labels'

    selection_info = []
    for f in os.listdir(LABEL_FOLDER):
        if f.find('video_label_single_') >= 0:
            selection_info +=  load_turker_video_labels(os.path.join(LABEL_FOLDER, f))

    queries = list(set([x['query_str'] for x in selection_info]))

    videos = []
    videos += [x['va'] for x in selection_info]
    videos += [x['vb'] for x in selection_info]
    videos = list(set(videos))
    # extract features
    video_info = {}
    FEATURE_FILE = 'features.pickle'
    #LOG_FILE = 'bad_videos.log'
    #log_fh = open(LOG_FILE, 'w')
    if os.path.exists(FEATURE_FILE):
        with open(FEATURE_FILE) as fh:
           video_info = pickle.load(fh)

        for s in selection_info:
            query = s['query_str']
            video_s_e = s['va']
            key = video_s_e + '-' + query
            if key not in video_info.keys(): 
                video_name, start_fid, end_fid = remove_start_end_fid(video_s_e)
                rcnn_bbx_list, rcnn_bbx_dict = load_video_rcnn_bbx('/home/t-yuche/ranking/gen_rank_data/rcnn-bbx-all', video_name) 
            
                dummy, w, h = get_video_fps(video_name)
                video_chunk = rcnn_bbx_list[int(start_fid): int(end_fid)]
                status, chunk_info = check_video_chunk(video_chunk, w, h, query)
                if len(chunk_info) == 0:
                    log_fh.write(query + ',' + video_s_e + '\n')
                if len(chunk_info) > 0:
                    video_info[key] = {'dwell_time': chunk_info['dwell_time'], 'obj_size': chunk_info['obj_size'][0] * chunk_info['obj_size'][1], 'moving_dist': chunk_info['moving_lvl']}
            
            video_name_s_e = s['vb']
            key = video_name_s_e + '-' + query
            if key not in video_info.keys(): 
                video_name, start_fid, end_fid = remove_start_end_fid(video_name_s_e)
                rcnn_bbx_list, rcnn_bbx_dict = load_video_rcnn_bbx('/home/t-yuche/ranking/gen_rank_data/rcnn-bbx-all', video_name) 
            
                dummy, w, h = get_video_fps(video_name)
                video_chunk = rcnn_bbx_list[int(start_fid): int(end_fid)]
                status, chunk_info = check_video_chunk(video_chunk, w, h, query)
                if len(chunk_info) == 0:
                    log_fh.write(query + ',' + video_name_s_e + '\n')
                if len(chunk_info) > 0:
                    video_info[key] = {'dwell_time': chunk_info['dwell_time'], 'obj_size': chunk_info['obj_size'][0] * chunk_info['obj_size'][1], 'moving_dist': chunk_info['moving_lvl']}
 
        with open(FEATURE_FILE, 'wb') as fh:
            pickle.dump(video_info, fh)
    
    #log_fh.close() 
    #queries = ['person', 'dog', 'car']
    queries = ['dog']
    for query in queries:
        obj_selection_infos = [x for x in selection_info if x['query_str'] == query]
        show_obj_size(video_info,  obj_selection_infos, query)
        show_dwell_time(video_info, obj_selection_infos, query)
        show_moving_dist(video_info, obj_selection_infos, query)
