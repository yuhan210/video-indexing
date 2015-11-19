import numpy as np


def get_video_rel(gt_video_rank, score_thresh, method = 2):
    """
    method 0: [1. 0.6309, 0.5, 0.4307, ...]
    method 1: reciprocal rank score
    method 2: cos_sim
    """
   
    video_rel = {}
    for tid, tup in enumerate(gt_video_rank):
        video_name = tup[0] 
        score = tup[1]
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
        score = tup[1]
    
        if score > score_thresh:
            if tid == 0:
                dcg_score += video_rel[video_name]
            else:
                dcg_score += video_rel[video_name]/(np.log2(tid + 1))

    return dcg_score
