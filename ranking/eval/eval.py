import os
import json
import pickle
from utils import *
from ndcg import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#try:
#    plt.style.use('ggplot')
#except:
#    pass

font = {'family' : 'normal',
        'size'   : 14,
    }
text = {'color': 'black'}
line = {'linewidth': 3}
matplotlib.rc('font', **font)
matplotlib.rc('text', **text)
matplotlib.rc('lines', **line)

VIDEOS = open('/mnt/video_list.txt').read().split()
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
 
class SubsampleSearch():  

    def __init__(self, ALL_TEXT_FOLDER = '/home/t-yuche/ranking/gen_rank_data/optimal-index'):

        self.optimal_data = {}
        for video_name in VIDEOS:
            with open(os.path.join(OPTIMAL_INPUT_FOLDER, video_name + '_' + str(0.5) + '.pickle')) as fh:
                self.optimal_data[video_name] = pickle.load(fh)

        self.vis_data = {}
        for sid in xrange(1):
            with open('/home/t-yuche/ranking/gen_rank_data/vis-log/all_vis_' + str(sid) + '.pickle') as fh:
                self.vis_data[sid] = pickle.load(fh)


    def search_text_vis(self, rand_idx, query_str, video_list):

        query_segs = query_str.split()
        query_dict = {}
        for word in query_segs:
            query_dict[word] = 1/float(len(query_segs))

        scores = {}
        for video_info in video_list:   
            video_name = video_info['video_name']
            start_fid = video_info['start_fid']
            end_fid = video_info['end_fid']
            
            key = str(start_fid) + '-' + str(end_fid)
            video_text = self.optimal_data[video_name][key] 
            text_score = getCosSimilarty(video_text, query_dict)    
            vis_score = visual_match(query_str, self.vis_data[rand_idx][video_name]['vis'])
 
            scores[video_name] = 0.5 * text_score + 0.5 * vis_score

        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_scores
    
    def search_text(self, rand_idx, query_str, video_list):

        query_segs = query_str.split()
        query_dict = {}
        for word in query_segs:
            query_dict[word] = 1/float(len(query_segs))

        scores = {}
        query_segs = query_str.split()
        for video_info in video_list:   
            video_name = video_info['video_name']
            start_fid = video_info['start_fid']
            end_fid = video_info['end_fid']
            
            key = str(start_fid) + '-' + str(end_fid)
            video_text = self.optimal_data[video_name][key] 
            text_score = getCosSimilarty(video_text, query_dict)    
 
            scores[video_name] = text_score

        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_scores
   
class OptimalSearch():  

    def __init__(self, ALL_TEXT_FOLDER = '/home/t-yuche/ranking/gen_rank_data/optimal-index'):

        self.optimal_data = {}
        for video_name in VIDEOS:
            with open(os.path.join(OPTIMAL_INPUT_FOLDER, video_name + '_' + str(0.5) + '.pickle')) as fh:
                self.optimal_data[video_name] = pickle.load(fh)

        self.vis_data = {}
        for sid in xrange(1):
            with open('/home/t-yuche/ranking/gen_rank_data/vis-log/all_vis_' + str(sid) + '.pickle') as fh:
                self.vis_data[sid] = pickle.load(fh)


    def search_text_vis(self, rand_idx, query_str, video_list):

        query_segs = query_str.split()
        query_dict = {}
        for word in query_segs:
            query_dict[word] = 1/float(len(query_segs))

        scores = {}
        for video_info in video_list:   
            video_name = video_info['video_name']
            start_fid = video_info['start_fid']
            end_fid = video_info['end_fid']
            
            key = str(start_fid) + '-' + str(end_fid)
            video_text = self.optimal_data[video_name][key] 
            text_score = getCosSimilarty(video_text, query_dict)    
            vis_score = visual_match(query_str, self.vis_data[rand_idx][video_name]['vis'])
 
            scores[video_name] = 0.5 * text_score + 0.5 * vis_score

        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_scores
    
    def search_text(self, rand_idx, query_str, video_list):

        query_segs = query_str.split()
        query_dict = {}
        for word in query_segs:
            query_dict[word] = 1/float(len(query_segs))

        scores = {}
        query_segs = query_str.split()
        for video_info in video_list:   
            video_name = video_info['video_name']
            start_fid = video_info['start_fid']
            end_fid = video_info['end_fid']
            
            key = str(start_fid) + '-' + str(end_fid)
            video_text = self.optimal_data[video_name][key] 
            text_score = getCosSimilarty(video_text, query_dict)    
 
            scores[video_name] = text_score

        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_scores

 
class MetadataSearch():

    def __init__(self, METADATA_FOLDER = '/mnt/video-info'):

        self.metadata = {}
       
        for stream_name in VIDEOS:
            with open(os.path.join(METADATA_FOLDER, stream_name + '.json')) as fh:
                data = json.load(fh)
                self.metadata[stream_name] = data 


    def search_metadata(self, query_str, video_list):

        scores = {}
        for stream_name in video_list:
            #print stream_name, self.metadata[stream_name]
            score, view_count = self.compute_metadata_score(self.metadata[stream_name], query_str)
            scores[stream_name] = (score, view_count)


        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse =True)
        title_scores = list(set(map(lambda x: x[1][0], sorted_scores)))
        title_scores.sort(reverse = True)
    
        sorted_viewcount = []
        for title_score in title_scores:
            scs = filter(lambda x: x[1][0] == title_score, sorted_scores)    
            scs = sorted(scs, key=lambda x:x[1][1], reverse =True)
            sorted_viewcount += scs

        return sorted_scores

    def get_score(self, value, edges):
        
        for i in xrange(len(edges) - 1):
            if value >= edges[i] and value <= edges[i + 1]: 
                return (i+1)/((len(edges) - 1) * 1.0)

    def get_type_score(self, value, value_type, TYPE = 0):

        if TYPE == 0:
            ratings = [0.0, 4.42857122421, 4.78807926178, 4.90198802948, 4.97389554977,5.0]
            likes = [0, 6, 91, 650, 4002, 740417]
            dislikes = [209830, 186, 30, 4, 0, 0]
            viewcounts = [0, 1200, 20949, 160330, 1194433, 591549856]
        
            if value_type == 'viewcount': 
                return self.get_score(value, viewcounts)
            elif value_type == 'rating': 
                return self.get_score(value, ratings)
            elif value_type == 'dislikes':
                for i in xrange(len(dislikes) - 1):
                    if value <= dislikes[i] and value >= dislikes[i+1]:
                        return (i+1)/((len(dislikes) -1) * 1.0)
            elif value_type == 'likes': 
                return self.get_score(value, likes)

        else:
            if value_type == 'viewcount': 
                return value
            elif value_type == 'rating': 
                return 0
            elif value_type == 'dislikes':
                return 0
            elif value_type == 'likes': 
                return 0
            

    def compute_metadata_score(self, stream_info, query_str): 
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
        metadata_features['rating'] = self.get_type_score(stream_info['rating'], 'rating', TYPE)
        
        #
        words = stream_info['description'].split()
        for q_w in query_words:
            for w in words:
                w = inflection.singularize(w.lower())
                if w.find(q_w) >= 0:
                    metadata_features['description'] += 1
                    break
        #
        metadata_features['viewcount'] = self.get_type_score(stream_info['viewcount'], 'viewcount', TYPE)
        #
        metadata_features['dislikes'] = self.get_type_score(stream_info['dislikes'], 'dislikes', TYPE)
        #
        metadata_features['likes'] = self.get_type_score(stream_info['likes'], 'likes', TYPE)

        #
        for q_w in query_words:        
            for w in stream_info['keywords']:
                w = inflection.singularize(w.lower())
                if w.find(q_w) >= 0:
                    metadata_features['keywords'] += 1
                    break
        
        # compute score
        score = 0
        if metadata_features['title'] and metadata_features['description'] and metadata_features['keywords']: 
            score = 15
            
        elif metadata_features['title'] or metadata_features['description'] or metadata_features['keywords']: 
            score = metadata_features['title'] + metadata_features['description'] + metadata_features['keywords']
            #for key in metadata_features.keys():
            #    score += metadata_features[key] 

        return score, metadata_features['viewcount']

def optimal_text_vis(stream_infos):
    sorted_opt_text_vis = sorted(stream_infos, key = lambda x: (x['vis_agg_info']['max_dwell_time']/150.) * x['optimal_score'], reverse = True)
    
    return [(x['video_name']+ '_' + str(x['start_fid']) + '_' + str(x['end_fid']) + '.mp4', x['optimal_score']) for x in sorted_opt_text_vis]

def optimal_text(stream_infos):
    sorted_opt_text = sorted(stream_infos, key = lambda x: x['optimal_score'], reverse = True)
 
    return [(x['video_name']+ '_' + str(x['start_fid']) + '_' + str(x['end_fid']) + '.mp4', x['optimal_score']) for x in sorted_opt_text]

def subsample_text_vis(stream_infos):
    sorted_opt_text_vis = sorted(stream_infos, key = lambda x: (x['vis_agg_info']['max_dwell_time']/150.) * x['subsample_score'], reverse = True)
    return [(x['video_name']+ '_' + str(x['start_fid']) + '_' + str(x['end_fid']) + '.mp4', x['subsample_score']) for x in sorted_opt_text_vis]
    

def subsample_text(stream_infos):
    sorted_opt_text = sorted(stream_infos, key = lambda x: x['subsample_score'], reverse = True)
 
    return [(x['video_name']+ '_' + str(x['start_fid']) + '_' + str(x['end_fid']) + '.mp4', x['subsample_score']) for x in sorted_opt_text]

def metadata_rank(stream_infos):

    title_scores  = list(set(map(lambda x: x['metadata_score'], stream_infos)))
    title_scores = sorted(title_scores, reverse = True)

    sorted_viewcount = []
    for title_score in title_scores:
        scs = filter(lambda x: x['metadata_score'] == title_score, stream_infos)    
        scs = sorted(scs, key=lambda x:x['metadata_viewcount'], reverse =True)
        sorted_viewcount += scs

   
    return [(x['video_name']+ '_' + str(x['start_fid']) + '_' + str(x['end_fid']) + '.mp4', x['metadata_viewcount']) for x in sorted_viewcount]

def get_gt_ranking(gt_ranking, query_infos):
    gt_ranking_w_score = []
    for r in gt_ranking:
        for info in query_infos: 
            if r.find(info['video_name']) >= 0:
                gt_ranking_w_score += [(r, info['gt_match_count'] )] 

    return gt_ranking_w_score

def get_spearman_coeff(gt_ranking, pred_ranking):

    ds = 0.0
    n = len(gt_ranking) * 1.0
    for gt_idx, gt_v in enumerate(gt_ranking):
        for pred_idx, pred_v in enumerate(pred_ranking): 
            if gt_v[0] == pred_v[0]:
                gt_r = len(gt_ranking) - gt_idx
                pred_r = len(pred_ranking) - pred_idx

                ds += ((gt_r - pred_r) ** 2)
    return  1 - ((ds * 6) / (n *(n**2 -1)))

METADATA = 'Metadata'
OPT_TXT = 'Text Only \n (No Subsampling)'
OPT_TXT_VIS = 'Text + \n Visual-Hint \n (No Subsampling)'
SUB_TXT = 'Text Only \n (Subsampling)'
SUB_TXT_VIS = 'Panorama'
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

    stream_rates = {}

    #for video_name in VIDEOS:
    #    fps, dummy, dummy = get_video_fps(video_name)
    #    stream_rates[video_name] = fps

    #with open('start_fid_set.pickle') as fh:
    #    start_fid_set = pickle.load(fh)


    labeled_queries = optimal_rank.keys()
    scores = {METADATA: [], OPT_TXT_VIS: [], OPT_TXT: [], SUB_TXT_VIS: [], SUB_TXT: []}
    for label_query in labeled_queries:
        query = label_query.split('-')[0]
        ssid = label_query.split('-')[1]

        query_infos = query_video_list[label_query]  
        metadata_ranking = metadata_rank(query_infos)
        optimal_ranking = optimal_rank[label_query]
        gt_ranking = get_gt_ranking(optimal_ranking, query_infos)
        gt_dcg = get_ranking_score(gt_ranking, gt_ranking)
        opt_text_ranking = optimal_text(query_infos)
        opt_text_vis_ranking = optimal_text_vis(query_infos)
        sub_text_ranking = subsample_text(query_infos)
        sub_text_vis_ranking = subsample_text_vis(query_infos)

        '''
        for r in optimal_ranking:
            for info in query_infos: 
                if r.find(info['video_name']) >= 0:
                    print info
        '''
        #print optimal_ranking
        #print metadata_ranking
        print label_query
        metadata_score = get_ranking_score(gt_ranking, metadata_ranking)/gt_dcg 
        opt_txt_score = get_ranking_score(gt_ranking, opt_text_ranking)/gt_dcg 
        opt_txt_vis_score = get_ranking_score(gt_ranking, opt_text_vis_ranking)/gt_dcg 
        sub_txt_score = get_ranking_score(gt_ranking, sub_text_ranking)/gt_dcg 
        sub_txt_vis_score = get_ranking_score(gt_ranking, sub_text_vis_ranking)/gt_dcg 

        metadata_score = get_spearman_coeff(gt_ranking, metadata_ranking)
        opt_txt_score = get_spearman_coeff(gt_ranking, opt_text_ranking) 
        sub_txt_score = get_spearman_coeff(gt_ranking, sub_text_ranking)
        sub_txt_vis_score = get_spearman_coeff(gt_ranking, sub_text_vis_ranking)
        opt_txt_vis_score = get_spearman_coeff(gt_ranking, opt_text_vis_ranking)

        #print metadata_score
        #print opt_txt_score
        #print sub_txt_score
        scores[METADATA] += [metadata_score]
        scores[OPT_TXT] += [opt_txt_score]
        scores[OPT_TXT_VIS] += [opt_txt_vis_score]
        scores[SUB_TXT] += [sub_txt_score]
        scores[SUB_TXT_VIS] += [sub_txt_vis_score]
        

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
    plt.show()
    
