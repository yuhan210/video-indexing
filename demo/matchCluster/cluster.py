import copy
import math
import json
import os
import sys
import math
import numpy as np

def getCosSimilarty(a_dict, b_dict):
    
    space = list(set(a_dict.keys()) | set(b_dict.keys()))
    
    # compute consine similarity (a dot b/ |a| * |b|)
    sumab = 0.0
    sumaa = 0.0
    sumbb = 0.0

    for dim in space:

        a = 0.0
        b = 0.0
        if dim in a_dict:
            a = a_dict[dim]
        if dim in b_dict:
            b = b_dict[dim]

        sumab += a * b
        sumaa += a * a
        sumbb += b * b        
    
    return sumab/(math.sqrt(sumaa * sumbb))

'''
Input: video_name
Output: term frequency for each frame (a list of (frame_number, tf) pairs)
'''
def load_turker_labels(video_name, turker_label_folder = "/home/t-yuche/gt-labeling/turker-labels"):

    video_folder = os.path.join(turker_label_folder, video_name)
    fs = sorted(os.listdir(video_folder), key=lambda key: int(key.split('.')[0]))

    all_nodes = [] 
    for frame_name in fs:
        
        label_path = os.path.join(video_folder, frame_name)
        labels = json.load(open(label_path))['gt_labels']
    
        #TODO: consider hierarchical words         
        ws = {}
        for choices in labels:
            for choice in choices:
                if choice not in ws:
                    ws[choice] = 1
                else:
                    ws[choice] += 1
        all_nodes += [(int(frame_name.split('.')[0]) ,ws)]

    return all_nodes




def match_indexes(indexes, query_str, portion_min_length = 15): # video_length is in secs
    '''
    Return a list of tuples (video_name, best_node, score)
    '''
    
    matched_nodes = []    

    for video_name in indexes:
        best_node, score = match_index(indexes[video_name], query_str, portion_min_length)
        matched_nodes += [(video_name, best_node, score)]
    
    return matched_nodes


def match_index(clusters, query_str, portion_min_length = 5): # video_length is in secs
    
    query_tf = {}
    for w in query_str.split(' '):
        if w not in query_tf:
            query_tf[w] = 1
        else:
            query_tf[w] += 1

    # TODO: make sure the following expression is true
    if len(clusters) == 0:
        return None, -1
 
    portion_min_length = min(portion_min_length * 30, max(clusters[-1]['n_idx']) - min(clusters[-1]['n_idx']))  

    dfs_sims = [] 
    # DFS    
    cur_node = clusters[-1]
    dfs_sims += [getCosSimilarty(query_tf, cur_node['tf'])]

    while (len(cur_node['descs']) > 0) or (max(cur_node['n_idx']) - min(cur_node['n_idx']) > portion_min_length):

        a_node = clusters[cur_node['descs'][0]]
        b_node = clusters[cur_node['descs'][1]]

        a_sim = getCosSimilarty(query_tf, a_node['tf'])
        b_sim = getCosSimilarty(query_tf, b_node['tf'])
        #print a_sim, b_sim, dfs_sims

        b_length = max(b_node['n_idx']) - min(b_node['n_idx'])
        a_length = max(a_node['n_idx']) - min(a_node['n_idx'])

        max_sim = max(a_sim, b_sim)
        if max_sim < dfs_sims[-1]:
            break
        
        node = b_node
        length = b_length
        if a_sim > b_sim:
            node = a_node
            length = a_length
        
        if length < portion_min_length:
            break

        dfs_sims += [max_sim]

        cur_node = node

    return cur_node, getCosSimilarty(query_tf, cur_node['tf'])



