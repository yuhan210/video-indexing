
from utils import *
def get_combined_tfs(tfs_dict):

    combined_tfs = {}
    # normalize
    deno = 0.0
    for d in tfs_dict:
        deno += 1
        for w in d['tf']:
            if w not in combined_tfs:
                combined_tfs[w] = 1
            else:
                combined_tfs[w] += 1

    for w in combined_tfs:
        combined_tfs[w] /= (deno * 1.0) 
    return combined_tfs


def detailed_measure(all_tf, subsampled_tf):
    match_count = 0
    for w in all_tf:
        if w in subsampled_tf:
            match_count += 1
    if len(all_tf) == 0:
        return -1
    return match_count/(len(all_tf) * 1.0)
    


def combine_all_modeldicts(_vgg_data, _msr_data, _rcnn_data, frame_paths):

    stop_words = get_stopwords()
    wptospd = word_pref_to_stopword_pref_dict()
    convert_dict = convert_to_equal_word()

    tf_list = []

    for frame_path in frame_paths:

        frame_name = frame_path.split('/')[-1]
        rcnn_data = _rcnn_data[frame_path]
        vgg_data = _vgg_data[frame_path]
        msr_data = _msr_data[frame_path]
   

        # combine words
        
        rcnn_ws = []
        if len(rcnn_data) > 0:
            for rcnn_idx, word in enumerate(rcnn_data['pred']['text']):
                ## the confidence is higher than 10^(-3) and is not background
                if rcnn_data['pred']['conf'][rcnn_idx] > 0.0005 and word not in stop_words:
                    rcnn_ws += [word]

        vgg_ws = []
        if len(vgg_data) > 0:
            for vgg_idx, w in enumerate(vgg_data['pred']['text']):
                w = wptospd[w]
                if w in convert_dict:
                    w = convert_dict[w]
                prob = (-1)*vgg_data['pred']['conf'][vgg_idx]
                if w not in stop_words and prob > 0.01:
                    vgg_ws += [w]

        
        msr_ws = [] 
        if len(msr_data) > 0:
            for msr_idx, w in enumerate(msr_data['words']['text']):
                w = inflection.singularize(w)

                prob = msr_data['words']['prob'][msr_idx]
                if w in convert_dict:
                    w = convert_dict[w]
                if w not in stop_words and len(w) != 0 and prob > -5 and msr_idx < 30:
                    msr_ws += [w]

        words = {}
        for w in rcnn_ws:
            if w not in words:
                words[w] = 1
            else:
                words[w] += 1
        for w in vgg_ws:
            if w not in words:
                words[w] = 1
            else:
                words[w] += 1
    
        for w_idx, w in enumerate(msr_ws):
            if w not in words:
                words[w] = 1
            else:
                words[w] += 1

        if '' in words:
            words.pop('', None)

        tf_list += [{'frame_name': frame_name, 'tf': words}]

    return tf_list




def get_inrange_fids(start_fid, end_fid, subsampled_frames):

    in_range_fids = []
 
    for f_count, f_name in enumerate(subsampled_frames):
        fid = int(f_name.split('.')[0])
               
        if fid >= start_fid and fid < end_fid:
            in_range_fids += [fid]
 
    if len(in_range_fids) == 0:
        for f_count, f_name in enumerate(subsampled_frames):
            fid = int(f_name.split('.')[0])
            
            if f_count == len(subsampled_frames)-1:
                if fid < start_fid:
                    in_range_fids += [fid]
                    break
            elif fid < start_fid and int(subsampled_frames[f_count + 1].split('.')[0]) >= end_fid:
                    in_range_fids += [fid]
                    break

    return in_range_fids 



def load_all_data_first_layer():

    INPUT_TRAIN_DATA = "/home/t-yuche/admission-control/train/train.data"
    videos = open('/mnt/video_list.txt').read().split()
    lines = [x.strip() for x in open(INPUT_TRAIN_DATA).readlines()]

    train_data = {}
    for video in videos:
        train_data[video] = {'0': [], '1': []}
    
    prev_fid = 0 
    prev_video_name = ''
    for line in lines: 
        segs = line.split(',')
        video_name = segs[0]
        if video_name != prev_video_name:
            prev_fid = 0
            prev_video_name = video_name
        fid = segs[1]
        if segs[2] == 'P':
            enctype = '0'
        else:
            enctype = '1'
        w = segs[3]
        h = segs[4]
                
         
        encsize = segs[5]
        
        mvsize = segs[6]
        meanmv = segs[7]
        maxmv = segs[8]
        minmv = segs[9]
        sobel = segs[10]
        illu = segs[11] 
        framediff = segs[12]
        phash = segs[13]
        colorhist = segs[14]
        siftmatch = segs[15]
        if segs[15] == '-1':
            siftmatch = '0'
        surfmatch = segs[16]
        if segs[16] == '-1':
            surfmatch = '0'
        label = segs[17]
        encsize_norm = float(encsize)/ (float(w)*float(h))
        
        dist_from_prevfid = (int(fid) - int(prev_fid))
        if int(label) == 1:
            prev_fid = int(fid)
 
        feature_tuple = [int(enctype), int(w), int(h), float(encsize), float(mvsize), float(meanmv), float(maxmv), float(minmv), int(dist_from_prevfid)] 

        train_data[video_name][label] += [feature_tuple]
        

    return train_data
