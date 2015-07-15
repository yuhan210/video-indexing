from tools.utils import load_video_recog, load_video_caption
from scipy.cluster.hierarchy import ward, dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from nltk.stem.wordnet import WordNetLemmatizer
import copy
import math
import nltk
import json
import os

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


def merge_verb_bog(window):

    window_verbtf = {}
    total_score = 0
    for caption in window:
        v_tf = caption['candidate']['verbs']
        print v_tf
        for v in v_tf:
            if v not in window_verbtf:
                window_verbtf[v] = v_tf[v]
            else:
                window_verbtf[v] += v_tf[v]

            total_score += v_tf[v]

    
    # normalize
    for v in window_verbtf:
        window_verbtf[v] /= (total_score * 1.0)

    print window_verbtf
    return window_verbtf

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
                # inversed (top 1 result gets 5 points)
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


'''
Find all the verbs in a sentence 
Input: string
Return: a list of verbs (in its simple tense) appeared in the sentence 
'''
def getVerbFromStr(sentence):

    segs = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(segs)
    
    verbs = []
    for word, tag in tags:
        if tag == 'VB' or tag == 'VBZ' or tag == 'VBN' or tag == 'VBD' or tag == 'VBG':
            verb_simple_tence = str(WordNetLemmatizer().lemmatize(word, 'v'))
            
            ## This is a hack. Remove 'be' verb  
            if verb_simple_tence != 'be': 
                verbs += [verb_simple_tence]

    return verbs


'''
Input: A list of strings
        method: 
            equal - does not consider the rank of the caption
            weighted - considers the rank (top 1 gets 5 points)
Output: Verb term frquency (verb to count mapping) 
''' 
def getVerbTfFromCaps(captions, method = 'equal'):
    #TODO: implement weighted version
  
    verb_tf = {}
    for sentence in captions:

        verbs = getVerbFromStr(sentence)
        for verb in verbs:
            if verb not in verb_tf:
                verb_tf[verb] = 1
            else:
                verb_tf[verb] += 1        

    return verb_tf 

def getCaptionVerbBatch(caption_data):


    for img_info in caption_data:
        captions = img_info['candidate']['text']
        verb_tf = getVerbTfFromCaps(captions)
        
        ## put the results back 
        img_info['candidate']['verbs'] = verb_tf 

    return caption_data

def getClusterKeyword(cluster_index, all_nodes):

    cluster_nodes = [all_nodes[idx] for idx in cluster_index]
    cluster_size = len(cluster_nodes)    
  
    cluster_recog = {}  
    cluster_verb = {}
    # get averaged term frequency
    for node in cluster_nodes:
        node_tf = node[1] # a dict with term: conf
        node_vtf = node[2]

        for term in node_tf:
            if term not in cluster_recog:
                cluster_recog[term] = node_tf[term]
            else:
                cluster_recog[term] += node_tf[term]
        
        for verb in node_vtf:
            if verb not in cluster_verb:
                cluster_verb[verb] = node_vtf[verb]
            else:
                cluster_verb[verb] += node_vtf[verb]
 
    # normalization
    for term in cluster_recog:
        cluster_recog[term] /= (cluster_size * 1.0)
    for verb in cluster_verb:
        cluster_verb[verb] /= (cluster_size * 1.0)
 
    # sort based on the prob
    cluster_recog = sorted(cluster_recog.items(), key=lambda x: x[1], reverse=True)
    cluster_verb = sorted(cluster_verb.items(), key=lambda x: x[1], reverse=True)
    
    return cluster_recog, cluster_verb

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
    
    return sumab/(math.sqrt(sumaa)  * math.sqrt(sumbb))


def getTimeDist(a_ts, b_ts, video_length):

    # linear   
    if a_ts > b_ts:
        return (a_ts - b_ts)/(video_length * 1.0)

    return (b_ts - a_ts)/(video_length * 1.0)


def getNodeDist(a_node, b_node, video_length, w_rtf = 0.3, w_vtf = 0.2, w_ts = 0.5):

    dist = w_rtf * (1 - getCosSimilarty(a_node[1], b_node[1])) + \
           w_vtf * (1 - getCosSimilarty(a_node[2], b_node[2])) + \
           w_ts * getTimeDist(a_node[0], b_node[0], video_length)
    #print (1 - getCosSimilarty(a_node[1], b_node[1])), getTimeDist(a_node[0], b_node[0], video_length), dist
    
    return dist
   
def selectClusterNum(linkage_matrix, cost_threshold = 0.5):

    costs = []
    n_clusters = []
    n_cluster = -1
    for i, dummy in enumerate(linkage_matrix):
        idx = len(linkage_matrix) - i - 1
        n_clusters += [i + 1]
        cost = linkage_matrix[idx, 2]
        costs += [cost]
                     
    #plt.figure(5)
    #plt.plot(n_clusters, costs)
    #plt.xlabel('# of Clusters')
    #plt.ylabel('Cluster merge cost')
    #plt.show()
    
    for idx, cost in enumerate(costs):
        if cost < cost_threshold:
            n_cluster = n_clusters[idx]
            break

    return n_cluster, n_clusters, costs


def load_turker_labels(video_name, turker_label_folder = "/home/t-yuche/gt-labeling/turker-labels"):

    video_folder = os.path.join(turker_label_folder, video_name)
    fs = sorted(os.listdir(video_folder), key=lambda key: int(key.split('.')[0]))

    all_nodes = [] 
    for frame_name in fs:
        
        label_path = os.path.join(video_folder, frame_name)
        labels = json.load(open(label_path))['gt_labels']

        print frame_name, labels 
        ws = {}
        for choices in labels:
            #print choices
            for choice in choices:
                if choice not in ws:
                    ws[choice] = 1
                else:
                    ws[choice] += 1
        all_nodes += [ws]

    return all_nodes, fs

def getGTNodeDist(a_node, b_node, a_framename, b_framename, length, w_word = 0.5, w_ts = 0.5):
    
    dist = w_word * (1 - getCosSimilarty(a_node, b_node)) + \
           w_ts * (getTimeDist(int(a_framename.split('.')[0]), int(b_framename.split('.')[0]), length))

    return dist 

def getPairwiseDist(all_nodes, frame_names):

    dist = [0.0 for x in range(len(all_nodes) * (len(all_nodes) - 1) / 2)]
    video_length = int(frame_names[-1].split('.')[0]) -  int(frame_names[0].split('.')[0])
    idx = 0
    for a_idx, a_node in enumerate(all_nodes):
        for b_idx in xrange(a_idx+1, len(all_nodes)):    
            dist[idx] = getGTNodeDist(a_node, all_nodes[b_idx], frame_names[a_idx], frame_names[b_idx], video_length)
            idx += 1
 
    return dist 



def getConstrainedPairwiseDist(clusters, pair_range = math.inf):
    
    darray = np.empty(len(clusters), len(clusters))
    darray.fill(np.inf)

    # for all the nodes that are active
    active_nodes = [x if x['active'] for x in clusters]

    # check nodes that are within range
    for a_idx, a_node in enumerate(active_nodes):
        for b_idx in xrange(len(active_nodes)):
                
            b_node = clusters[b_idx] 

            # TODO: other constraints? 
            within_range = False
            for i in a_node['r_idx']:
                for j in b_node['r_idx']:
                    if abs(i-j) <= pair_range:
                        within_range = True
                        break

            if not within_range:
                break
            dist = 1 - getCosSimilarty(a_node['tf'], b_node['tf']) 
            darray[a_node['idx'], b_node['idx']] = dist

    return darray 

def getActiveClusterNum(clusters):
    
    counter = 0
    for cluster in clusters:
        if cluster['active']:
            counter += 1

    return counter

def cluster(all_nodes, num_clusters = 1): 

    # data: a list of (frame_timestamp, tf)s, should be sorted by frame_timestamp
    clusters = []
    # create clusters
    for idx, node in enumerate(all_nodes):
        time_stamp = node[0] # int frame number
        tf = node[1] # {desk:1, table:2}
        c = {}
        c['idx'] = idx
        c['ts'] = [time_stamp]
        c['r_idx'] = [idx]
        c['tf'] = tf
        c['active'] = True
        clusters += [c]
    
    linkage_matrix = []
    while getActiveClusterNum(clusters) > max(1, num_clusters):
        
        # get pairwise distance
        darray = getConstrainedPairwiseDist(clusters, 1)              
  
        # merge the two closest clusters
        a_idx, b_idx = np.unravel_index(np.argmin(darray), darray.shape)
        dist = np.min(darray)

        # create new cluster
        # merge node a and b
        a_node = clusters[a_idx]
        b_node = clusters[b_idx] 

        c = {}
        c['idx'] = len(clusters)
        c['ts'] = (a_node['ts'] + b_node['ts']).sort()
        c['r_idx'] = (a_node['idx'] + b_node['idx']).sort()

        # merge two term freq
        tf = copy.deepcopy(a_node['tf'])
        for key in b_node:
            if key not in tf:
                tf[key] = b_node[key]
            else:
                tf[key] += b_node[key]

        c['tf'] = tf 
        c['active'] = True
        clusters += [c]
        
        # disable merged cluster
        clusters[a_idx]['active'] = False
        clusters[b_idx]['active'] = False

        # update cluster (linkage_matrix)  append [clusterID_1, clusterID_2, distance, # of observation in the new cluster]
        linkage_matrix.append([a_idx, b_idx, dist, len(c['r_idx'])])
            
    return clusters, np.array(linkage_matrix)

if __name__ == "__main__":

    '''
    video_name = sys.argv[1]
    recog_folder = sys.argv[2]
    '''
    #video_name = "warty_pigs_through_google_glass_Xs3x3lCl8_E" 
    #video_name = "yongpyong_resort_south_korea_snowboarding_with_google_glass__part_3_1LzvqaAu_Mk"
    #video_name = "beyonce__drunk_in_love__red_couch_session_by_dan_henig_a1puW6igXcg"
    #video_name = "thoroughbred_horse_through_googleglass_IbXdHo9CN1I"
    video_name = "kids_playing_basketball"
    recog_folder = "/home/t-yuche/frame-analysis/recognition"
    caption_folder = "/home/t-yuche/frame-analysis/caption"
    #n_clusters = 3

    # load labels from turkers
    gt_nodes, frames = load_turker_labels(video_name)
    gt_dist = getPairwiseDist(gt_nodes, frames) 
    
    linkage_matrix = linkage(gt_dist, method='average')
    print linkage_matrix
    exit(-1)
    n_clusters, cluster_list, costs = selectClusterNum(linkage_matrix)
    ax = dendrogram(linkage_matrix, orientation='right');
    fc = fcluster(linkage_matrix, n_clusters,'maxclust') 
    
    node_to_cluster = {}
    cluster_to_node = [[] for x in xrange(n_clusters)]
    for idx, fc in enumerate(fc):
        node_to_cluster[int(ax['ivl'][idx])] = fc
        cluster_to_node[fc-1] += [int(ax['ivl'][idx])]

    
    cluster_num = []
    for idx in xrange(len(gt_nodes)):
        cluster_num += [node_to_cluster[idx]]

    plt.figure(2)
    plt.plot(range(len(gt_nodes)), cluster_num)
    plt.xticks(range(len(gt_nodes)), frames, rotation = 'vertical')
    plt.xlabel('Frame Number')
    plt.ylabel('Cluster Number')
    plt.ylim([-1, n_clusters+1])
    
    plt.figure(3)
    plt.plot(cluster_list, costs, 'x')
    plt.xlabel('# of Clusters')
    plt.ylabel('Cluster merge cost')
    plt.show()
    exit(-1)

    recog_data = load_video_recog(recog_folder, video_name) 
    #recog_data = recog_data[0:200]

    caption_data = load_video_caption(caption_folder, video_name)
    #caption_data = caption_data[0:200]
    caption_data = getCaptionVerbBatch(caption_data)
    
    # on-line hierarchical clustering
    window_size = 5 # frames
    window = []
    prev_node = (-1, {}) 
    all_nodes = []

    sims = []
    index = []
    for idx, recog in enumerate(recog_data):
        caption = caption_data[idx]
        # print recog['conf'], recog['text'] 
        window += [(recog, caption)]
        if len(window) == window_size:
             
            node = (idx, merge_recogs_bog([x[0] for x in window]), merge_verb_bog([x[1] for x in window]))
            window = []
            print node 
            # compute similarity
            if prev_node[0] != -1:
                sim = 0.5 * getCosSimilarty(prev_node[1], node[1]) + 0.5 * getCosSimilarty(prev_node[2], node[2]) 
                sims += [sim]
                index += [idx]
            #print node
            all_nodes += [node]
            prev_node = node
            #print prev_merged
     
    #plt.plot(index, sims)
    #plt.show() 
    '''
    for node in all_nodes:
        print node
    '''
    # off-line wards hierarchical clustering
    video_length = max([x[0] for x in all_nodes]) - min([x[0] for x in all_nodes])
    dist = [0.0 for x in range(len(all_nodes) * (len(all_nodes) - 1) / 2)]
    idx = 0
    for a_idx, a_node in enumerate(all_nodes):
        for b_idx in xrange(a_idx+1, len(all_nodes)):    
            dist[idx] = getNodeDist(a_node, all_nodes[b_idx], video_length)
            idx += 1

    
    # clustering
    # average single complete
    linkage_matrix = linkage(dist, method='average')
    n_clusters = selectClusterNum(linkage_matrix)
    print n_clusters 
    ax = dendrogram(linkage_matrix, orientation='right', no_plot='True');
    
    fc = fcluster(linkage_matrix, n_clusters,'maxclust') 
   
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
    '''
    for cluster_num, cluster in enumerate(cluster_to_node):
        keywords, keyverbs = getClusterKeyword(cluster, all_nodes)
        print '\nCluster %d' % (cluster_num + 1)
        for key in keywords:
            print '%s : %f' % (key[0], key[1])

        print ''
        for verb in keyverbs:
            print '%s : %f' % (verb[0], verb[1])
    '''
    '''  
    for chunk in cluster_to_node:
        print sorted(chunk)

    '''
    #for chunk in cluster_to_node:
    #    print min(chunk), max(chunk)

    cluster_num = []
    for idx in xrange(len(all_nodes)):
        cluster_num += [node_to_cluster[idx]]

    plt.figure(2)
    plt.plot(range(len(all_nodes)), cluster_num)
    plt.xticks(range(len(all_nodes)), [x[0] for x in all_nodes], rotation = 'vertical')
    plt.xlabel('Frame Number')
    plt.ylabel('Cluster Number')
    plt.ylim([-1, n_clusters+1])
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
  
    
