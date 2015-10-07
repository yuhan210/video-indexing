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

        if vid == 50:
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


def get_inrange_fids(start_fid, end_fid, subsampled_frames):

    prev_fid = -1
    in_range_fids = []

    for f_count, f_name in enumerate(subsampled_frames):
        fid = int(f_name.split('.')[0])

        if f_count == 0:
            prev_fid = fid
                
        if prev_fid < start_fid and fid > end_fid:
            in_range_fids += [prev_fid]
            break
               
        if fid >= start_fid and fid <= end_fid:
            in_range_fids += [fid]
 
        if fid > end_fid:
            break

    return in_range_fids 




def get_subsampled_tf(video_name, vgg_data, msr_data, rcnn_data, fei_data, in_range_fids):
    s_vgg_data = []
    s_fei_data = []
    s_msr_data = []
    s_rcnn_data = []
    
    for fid in in_range_fids:

        s_vgg_data += filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == fid , vgg_data)
        s_fei_data += filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == fid , fei_data)
        s_msr_data += filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == fid , msr_data)
        s_rcnn_data += filter(lambda x: int(x['image_path'].split('/')[-1].split('.')[0]) == fid, rcnn_data)

    s_tf_list = combine_all_models(video_name, s_vgg_data, s_msr_data, s_rcnn_data, s_fei_data)
    s_tf = get_combined_tfs(s_tf_list)

    return s_tf

def run(vgg, fei, msr, rcnn):
    
    SERVER_STORAGE_FRAMES = 5 * 30 # 5 sec * 30 fps

    QUERY_LIST = './query.txt'
    queries = open(QUERY_LIST).read().split()
    
    VIDEO_LIST = '/mnt/video_list.txt'
    videos = open(VIDEO_LIST).read().split()


    ## Modify ##
    greedy_folder = '/home/t-yuche/luckyframe-eval/greedy-log'
    ##    

    server_storage = []
    for query in queries:
        n_video_tfs = []
        uni_video_tfs = []
        greedy_video_tfs = []

        for vid, video_name in enumerate(videos):
            
            if video_name not in vgg:
                continue
            
            # load data
            vgg_data = vgg[video_name]
            fei_data = fei[video_name]
            msr_data = msr[video_name]
            rcnn_data = rcnn[video_name]

            # load greedy subsampled frames
            greedy_gt_path = os.path.join(greedy_folder, video_name +  '_gtframe.pickle')
            with open(greedy_gt_path) as gt_fh:
                selected_frame_obj = pickle.load(gt_fh)
                greedy_frames = [ str(x) + '.jpg' for x in selected_frame_obj['picked_f']]
                video_len_f = selected_frame_obj['total_frame']
                subsampled_rate = selected_frame_obj['picked_rate']
                 
            # baseline
            uniform_frames = naive_subsample_frames(os.listdir(os.path.join('/mnt/frames', video_name)), subsampled_rate) 
            

            # add random delays to each video stream -- starting at random position
            hash_str = video_name + query
            hash_val = int(abs(hash(hash_str)))%10
            video_start_fid = (video_len_f/10) * hash_val  
            video_end_fid = max(video_start_fid + SERVER_STORAGE_FRAMES, video_len_f - 1)
          
            ## Non-subsampled video ## 
            # process in-server info for non-subsampled video
            _vgg_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) >= video_start_fid and int(x['img_path'].split('/')[-1].split('.')[0]) <= video_end_fid, vgg_data)
            _fei_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) >= video_start_fid and int(x['img_path'].split('/')[-1].split('.')[0]) <= video_end_fid, fei_data)
            _msr_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) >= video_start_fid and int(x['img_path'].split('/')[-1].split('.')[0]) <= video_end_fid, msr_data)
            _rcnn_data = filter(lambda x: int(x['image_path'].split('/')[-1].split('.')[0]) >= video_start_fid and int(x['image_path'].split('/')[-1].split('.')[0]) <= video_end_fid, rcnn_data)
            
            tf_list = combine_all_models(video_name, _vgg_data, _msr_data, _rcnn_data, _fei_data)
            tf = get_combined_tfs(tf_list)

            #print tf
            n_video_tfs += [{'video_name': video_name, 'tf': tf}]
    
            ## Baseline -- uniformly subsampled video ## 
            # process in-server info for subsampled video
            uni_range_fids = get_inrange_fids(video_start_fid, video_end_fid, uniform_frames)
            uni_tf = get_subsampled_tf(video_name, vgg_data, msr_data, rcnn_data, fei_data, uni_range_fids)
            uni_video_tfs += [{'video_name': video_name, 'tf': uni_tf}]

            ## Greedy -- subsampled video ##
            greedy_range_fids = get_inrange_fids(video_start_fid, video_end_fid, greedy_frames)
            greedy_tf = get_subsampled_tf(video_name, vgg_data, msr_data, rcnn_data, fei_data, greedy_range_fids)
            greedy_video_tfs += [{'video_name': video_name, 'tf': greedy_tf}]


        # ranking for a given query
        n_video_rank = rank(query, n_video_tfs)
        uni_video_rank = rank(query, uni_video_tfs) 
        greedy_video_rank = rank(query, greedy_video_tfs)
 
        print n_video_rank                        
        print uni_video_rank
        print greedy_video_rank    

#def run(vgg, fei, msr, rcnn):
#    return 
