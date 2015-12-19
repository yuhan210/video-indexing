import numpy as np

def get_Spearmansfr_score_tuplist(gt_video_rank, pred_video_rank):

    video_names = [x[0] for x in gt_video_rank]   
    s = len(video_names)

    fr = 0.0    
    for video_name in video_names:
        gt_rank_pos = [idx for idx, x in enumerate(gt_video_rank) if x[0] == video_name][0]
        pred_rank_pos = [idx for idx, x in enumerate(pred_video_rank) if x[0] == video_name][0]
        fr += abs(gt_rank_pos - pred_rank_pos) 

    return (1.0 - ((2* fr) / ((s ** 2) * 1.0)))

def get_Spearmansfr_score(gt_video_rank, pred_video_rank):

    s = len(gt_video_rank)

    fr = 0.0    
    for video_name in gt_video_rank:
        gt_rank_pos = s - gt_video_rank.index(video_name) 
        pred_rank_pos = s - pred_video_rank.index(video_name) 
        diff = abs(gt_rank_pos - pred_rank_pos) 
        fr += abs(gt_rank_pos - pred_rank_pos) 

    return (1.0 - ((2* fr) / ((s ** 2) * 1.0)))
    
def get_Spearmansfr_intersect_score(gt_video_rank, pred_video_rank, intersect_video, scheme):

    s = len(intersect_video)

    # remove non-exist videos
    tmp_vs = list(gt_video_rank)
    for v in tmp_vs:
        if v not in intersect_video:
            gt_video_rank.remove(v)
    tmp_vs = list(pred_video_rank)
    for v in tmp_vs:
        if v not in intersect_video:
            pred_video_rank.remove(v)

    print scheme + ':'
    print pred_video_rank

    fr = 0.0    
    for video_name in gt_video_rank:
        gt_rank_pos = s - gt_video_rank.index(video_name) 
        pred_rank_pos = s - pred_video_rank.index(video_name) 
        diff = abs(gt_rank_pos - pred_rank_pos) 
        fr += abs(gt_rank_pos - pred_rank_pos) 

    return (1.0 - ((2* fr) / ((s ** 2) * 1.0)))

def get_ranking_score(gt_video_rank, pred_video_rank):
    
    video_rel = get_video_rel(gt_video_rank)
    return dcg(video_rel, pred_video_rank)


def get_video_rel(gt_video_rank, score_thresh = 0, method = 1):
    """
    method 0: [1. 0.6309, 0.5, 0.4307, ...]
    method 1: reciprocal rank score
    method 2: cos_sim
    """
   
    video_rel = {}
    for tid, tup in enumerate(gt_video_rank):
        video_name = tup[0]
        score = 1
        if method == 0:
            if score > score_thresh:
                video_rel[video_name] = 1/np.log2(tid + 2)
            else:
                video_rel[video_name] = 0.0

        elif method == 1:
            if score > score_thresh: 
                video_rel[video_name] = 1/((tid + 1) * 1.0) 
            else:
                video_rel[video_name] = 0.0
        elif method == 2:
            if score > score_thresh:
                video_rel[video_name] = score
            else:
                video_rel[video_name] = 0.0
    
        
    return video_rel



def dcg(video_rel, test_video_rank, score_thresh = 0):
    """
    video_rel: mapping from video_name to its score
    test_video_rank: a sorted list of (video_name, score) 
    """
    dcg_score = 0.0

    for tid, tup in enumerate(test_video_rank):
        video_name = tup[0]
        score = 1
    
        if score > score_thresh:
            if tid == 0:
                dcg_score += video_rel[video_name]
            else:
                dcg_score += video_rel[video_name]/(np.log2(tid + 1))
                #dcg_score  += video_rel[video_name]/((tid+1) * 1.0)
                #video_rel[video_name] = 1/((tid + 1) * 1.0) 


    return dcg_score

a = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
b = ['j', 'b', 'd', 'c', 'f', 'e', 'g', 'h', 'i', 'a']
print get_Spearmansfr_score(a, b)
