from tools.utils import load_video_recog
from scipy.cluster.hierarchy import ward, dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform


import matplotlib
import matplotlib.pyplot as plt
try:
    plt.style.use('ggplot')
except:
    pass

import sys
import math
import numpy as np
import pickle

def load_labels():

    with open("/home/t-yuche/caffe/data/ilsvrc12/synset_words.txt") as f:
        labels_df = pd.DataFrame([
            {
                'synset_id':l.strip().split(' ')[0],
                'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
            }
            for l in f.readlines()
        ])
    labels = labels_df.sort('synset_id')['name'].values
    labels += ['others']
    return labels


'''
method:
    equal: does not consider ranking
    weighted: consider ranking
'''
def merge_recogs_bog(window, method='weighted'):

    window_recog = {}
     
    for idx, recog in enumerate(window):
        
        for rank, pred in enumerate(recog['pred']['text']):
            total_ranks = len(recog['pred']['text'])
            
            weight = 1

            # determine the weight based on method
            if method == 'equal':
                weight = 1                      

            elif method == 'weighted':
                # inversed top 1 result gets 5 points
                weight = total_ranks - rank

            if pred not in window_recog:
                window_recog[pred] = weight
            else:
                window_recog[pred] += weight

      
    # normalize
    total_count = sum([window_recog[key] for key in window_recog])
    for key in window_recog:
        window_recog[key] /= (total_count * 1.0)

    return window_recog

'''
My own term frequency invention
'''
def merge_recogs_top5softmax(window):
    
    window_recog = {'others': 0.0}

    for idx, recog in enumerate(window):
        recog =  recog['pred'] 

        ## uh. *(-1) cause we multiplied by -1 when getting softmax
        confs = [ (-1) * conf for conf in recog['conf']]
        window_recog['others'] += 1 - sum(confs)
        
        for idx, pred in enumerate(recog['text']):
            if pred not in window_recog:
                window_recog[pred] = (-1) * recog['conf'][idx] 
            else:
                window_recog[pred] += (-1) * recog['conf'][idx] 
        
    for key in window_recog:
        window_recog[key] /= len(window)

    '''
    out_str = ""    
    for key in window_recog:
        out_str += "%s: %0.4f "% (key, window_recog[key])
    print out_str
    '''
    return window_recog


def getClusterKeyword(cluster_index, all_nodes):

    cluster_nodes = [all_nodes[idx] for idx in cluster_index]
    cluster_size = len(cluster_nodes)    
  
    cluster_recog = {}  
    # get averaged term frequency
    for node in cluster_nodes:
        node_tf = node[1] # a dict with term: conf
        
        for term in node_tf:
            if term not in cluster_recog:
                cluster_recog[term] = node_tf[term]
            else:
                cluster_recog[term] += node_tf[term]
               
    # normalization
    for term in cluster_recog:
        cluster_recog[term] /= cluster_size
    
    # sort based on the prob
    cluster_recog = sorted(cluster_recog.items(), key=lambda x: x[1], reverse=True)
    
    return cluster_recog

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


def getTimeDist(a_ts, b_ts, video_length):

    # linear   
    if a_ts > b_ts:
        return (a_ts - b_ts)/(video_length * 1.0)

    return (b_ts - a_ts)/(video_length * 1.0)


def getNodeDist(a_node, b_node, video_length, w_tf = 0.4, w_ts = 0.6):

    dist = w_tf * (1 - getCosSimilarty(a_node[1], b_node[1])) + w_ts * getTimeDist(a_node[0], b_node[0], video_length)
    print (1 - getCosSimilarty(a_node[1], b_node[1])), getTimeDist(a_node[0], b_node[0], video_length), dist
    
    return dist
    

if __name__ == "__main__":

    '''
    video_name = sys.argv[1]
    recog_folder = sys.argv[2]
    '''
    #video_name = "warty_pigs_through_google_glass_Xs3x3lCl8_E" 
    video_name = "beyonce__drunk_in_love__red_couch_session_by_dan_henig_a1puW6igXcg"
    #video_name = "google_glass_films_google_glass__real_estate_tour_throughglass_7e5iDdPGeGA"
    recog_folder = "/home/t-yuche/frame-analysis/recognition"
    n_clusters = 2 


    recog_data = load_video_recog(recog_folder, video_name)
    recog_data = recog_data[0:200]


    # on-line hierarchical clustering
    window_size = 1 # frames
    window = []
    prev_node = (-1, {}) 
    all_nodes = []

    sims = []
    index = []
    for idx, recog in enumerate(recog_data):
        # print recog['conf'], recog['text'] 
        window += [recog]
        if len(window) == window_size:

            node = (idx, merge_recogs_bog(window))
            window = []
            
            # compute similarity
            if prev_node[0] != -1:
                sim = getCosSimilarty(prev_node[1], node[1]) 
                sims += [sim]
                index += [idx]
            #print node
            all_nodes += [node]
            prev_node = node
            #print prev_merged
    
    #plt.plot(index, sims)
    #plt.show() 
    print all_nodes 
    # off-line wards hierarchical clustering
    video_length = max([x[0] for x in all_nodes]) - min([x[0] for x in all_nodes])
    dist = [0.0 for x in range(len(all_nodes) * (len(all_nodes) - 1) / 2)]
    idx = 0
    for a_idx, a_node in enumerate(all_nodes):
        for b_idx in xrange(a_idx+1, len(all_nodes)):    
            dist[idx] = getNodeDist(a_node, all_nodes[b_idx], video_length)
            idx += 1

    
    # clustering 
    linkage_matrix = linkage(dist, method='average')
   
    ax = dendrogram(linkage_matrix, orientation='right');
    
    fc = fcluster(linkage_matrix, n_clusters, 'maxclust') 
   
    # assign nodes to clusters
    node_to_cluster = {}
    cluster_to_node = [[] for x in xrange(n_clusters)]
    for idx, fc in enumerate(fc):
        node_to_cluster[int(ax['ivl'][idx])] = fc
        cluster_to_node[fc-1] += [int(ax['ivl'][idx])]

    
    cluster_num = []
    for idx in xrange(len(all_nodes)):
        cluster_num += [node_to_cluster[idx]]
    
    # get the keywords for each cluster
    
    for cluster_num, cluster in enumerate(cluster_to_node):
        keywords = getClusterKeyword(cluster, all_nodes)
        print '\nCluster %d' % (cluster_num)
        for key in keywords:
            print '%s : %f' % (key[0], key[1])
 
    '''  
    for chunk in cluster_to_node:
        print sorted(chunk)

    for chunk in cluster_to_node:
        print min(chunk), max(chunk)

    '''
    cluster_num = []
    for idx in xrange(len(all_nodes)):
        cluster_num += [node_to_cluster[idx]]

    plt.figure(2)
    plt.plot(range(len(all_nodes)), cluster_num)
    plt.xticks(range(len(all_nodes)), [x[0] for x in all_nodes], rotation = 'vertical')
    plt.xlabel('Frame Number')
    plt.ylabel('Cluster Number')
    plt.show()
    '''
    fig, ax = plt.subplots(figsize=(15, 20)) # set size


    plt.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    '''
   # plt.tight_layout() #show plot with tight layout
   # plt.show()
  
    
