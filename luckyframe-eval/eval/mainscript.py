from utils import *
import operator
import pickle

def load():
    
    VIDEO_LIST = '/mnt/video_list.txt'
    videos = open(VIDEO_LIST).read().split()
    
    vgg = {}
    fei_cap = {}
    msr_cap = {}
    rcnn = {}

    for vid, video_name in enumerate(videos):
        _vgg_data = load_video_recog('/mnt/tags/vgg-classify-all', video_name)
        _fei_caption_data = load_video_caption('/mnt/tags/fei-caption-all', video_name)
        _msr_cap_data = load_video_msr_caption('/mnt/tags/msr-caption-all', video_name)
        _rcnn_data = load_video_rcnn('/mnt/tags/rcnn-info-all', video_name)
        
        vgg[video_name] = _vgg_data
        fei_cap[video_name] = _fei_caption_data
        msr_cap[video_name] = _msr_cap_data
        rcnn[video_name] = _rcnn_data

        if vid == 400:
            break
    return vgg, fei_cap, msr_cap, rcnn




def load_subsampled_frames(video_name, folder):

    with open(os.path.join(folder, video_name)) as fh:
        keyframes = sorted(pickle.load(fh), key = lambda x: int(x.split('.')[0]))

    return keyframes

def rank(query, video_tfs):
   
    # 
    query_dict = {}
    query_dict[query] = 1

    video_scores = {}
    for video_d in video_tfs:
        video_name = video_d['video_name']
        video_tf = video_d['tf']
        cos_sim = cos_similarty(query_dict, video_tf)  
        video_scores[video_name] = cos_sim

    ranked_video = sorted(video_scores.items(), key = operator.itemgetter(1), reverse=True)

    return ranked_video

def run(vgg, fei, msr, rcnn):
    
    SERVER_STORAGE_FRAMES = 5 * 30 # 5 sec * 30 fps

    QUERY_LIST = './query.txt'
    queries = open(QUERY_LIST).read().split()
    
    VIDEO_LIST = '/mnt/video_list.txt'
    videos = open(VIDEO_LIST).read().split()


    ## Modify ##
    subsample_folder = './train/log'
    ##    

    server_storage = []
    for query in queries:
        n_video_tfs = []
        s_video_tfs = []
        for video_name in videos:
        
            if video_name not in vgg: 
                continue

            #subsampled_frames = load_subsampled_frames(video_name, subsample_folder)
            #naive_subsample_frames(all_frames, RETAINED_RATE) 
            video_len_f = len(vgg[video_name])
            
            # add random delays to each video stream -- starting at random position
            hash_str = video_name + query
            hash_val = int(abs(hash(hash_str)))%10
            video_start_fid = (video_len_f/10) * hash_val  
            video_end_fid = max(video_start_fid + SERVER_STORAGE_FRAMES, video_len_f - 1)
           
            # process in-server info for non-subsampled video
            _vgg_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) >= video_start_fid and int(x['img_path'].split('/')[-1].split('.')[0]) <= video_end_fid, vgg[video_name])
            _fei_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) >= video_start_fid and int(x['img_path'].split('/')[-1].split('.')[0]) <= video_end_fid, fei[video_name])
            _msr_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) >= video_start_fid and int(x['img_path'].split('/')[-1].split('.')[0]) <= video_end_fid, msr[video_name])
            _rcnn_data = filter(lambda x: int(x['image_path'].split('/')[-1].split('.')[0]) >= video_start_fid and int(x['image_path'].split('/')[-1].split('.')[0]) <= video_end_fid, rcnn[video_name])
            
            tf_list = combine_all_models(video_name, _vgg_data, _msr_data, _rcnn_data, _fei_data)
            tf = get_combined_tfs(tf_list)

            #print tf
            n_video_tfs += [{'video_name': video_name, 'tf': tf}]
            
            # process in-server info for subsampled video
            # can only be a subset of the non-subsampled one/the previous one (not in this range)
                  

        # ranking for a given query
        n_video_rank = rank(query, n_video_tfs)      
        print n_video_rank                        
    

#def run(vgg, fei, msr, rcnn):
#    return 
