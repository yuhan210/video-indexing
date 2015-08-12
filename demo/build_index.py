import sys
import pickle
CLUSTER_PATH = '/home/t-yuche/clustering/clusterLib'
sys.path.append(CLUSTER_PATH)
from cluster import *

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print 'Usage:', sys.argv[0], ' (0: reset 1: append) index_file video_list'
        exit(-1)
    
    IS_APPEND = int(sys.argv[1])
    index_file = sys.argv[2]
    video_list_file = sys.argv[3]
    video_list = open(video_list_file).read().splitlines()

    index = {}
    if IS_APPEND:
        with open(index_file, 'rb') as handle:
            prev_index = pickle.load(handle)
            print prev_index    

    for video_name in video_list:
        video_name = video_name.split('.')[0]
        print video_name 
        if IS_APPEND:
            if video_name in index:
                continue

        
        gt_nodes = load_turker_labels(video_name)
        clusters, linkage_matrix = cluster(gt_nodes)
        index[video_name] = clusters


    with open(index_file, 'wb') as handle:
        pickle.dump(index, handle)
        
