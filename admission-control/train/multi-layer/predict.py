import sys
sys.path.append('/home/t-yuche/admission-control/train/svm/code')
from svm_utils import *
from vision import *
import pickle
import sys
sys.path.append('/home/t-yuche/admission-control/train/svm/liblinear/python')
from svmutil import *
import sys
from utils import *
import numpy as np
from lib import *

ENC_TYPE = 1
IMG_WIDTH = 2
IMG_HEIGHT = 3
ENC_SIZE = 4
MV_SIZE = 5
MV_MEAN = 6
MV_MAX = 7
MV_MIN = 8
SOBEL = 9
ILLU = 10
FRAME_DIFF = 11
PHASH = 12
COLORHIST = 13
SIFTMATCH = 14
SURFMATCH = 15
DIST_FROM_PFID = 16

PRED_TRACELOG = './pred-trace'
MV_INPUT_FOLDER = '/home/t-yuche/admission-control/train/mv_log'
FST_MODELS = './1st_models'
SND_MODELS = '/home/t-yuche/admission-control/train/svm/code/models'
SCORE_OUTPUT_FOLDER = './scores'

def predict_video(video_name, enc_model, svm_model):
   
    n_frames = get_video_frame_num(video_name) 
    pred_trace = {'picked_f': [], 'total_frame': 0, 'picked_rate': 0.0}
 
    encdata = getMetadata(video_name) 
    cvdata = getCVInfoFromLog(video_name)
    mv_file = os.path.join(MV_INPUT_FOLDER, video_name + '.pickle')
    with open(mv_file) as fh:
        mv_features = pickle.load(fh)
    w = encdata['metadata']['w'] 
    h = encdata['metadata']['h']

    selected_fids = [0]
    prev_fid = 0
    for fid in xrange(1, n_frames):
        frame_name = str(fid) + '.jpg'
        prev_frame_name = str(prev_fid) + '.jpg'

 
        # generate features
        #### single frame features
        enc = encdata[frame_name]
        x = {}
        if enc['type'] == 'P': 
            x[ENC_TYPE] = 0
        else:
            x[ENC_TYPE] = 1

        cv = cvdata[frame_name]
        mv = mv_features[fid]
        #########################

        x[IMG_WIDTH] = w
        x[IMG_HEIGHT] = h
        x[ENC_SIZE] = enc['size']
        x[MV_SIZE] = mv[0] 
        x[MV_MEAN] = mv[1]
        x[MV_MAX] = mv[2]
        x[MV_MIN] = mv[3]
        x[DIST_FROM_PFID] = fid - prev_fid
 
        # predict if we should pick this frame
        fst_layer_f = [x[ENC_TYPE], x[IMG_WIDTH], x[IMG_HEIGHT], x[ENC_SIZE], x[MV_SIZE], x[MV_MEAN], x[MV_MAX], x[MV_MIN], x[DIST_FROM_PFID]]
        pred_y = enc_model.predict(fst_layer_f)
        if int(pred_y) == 0:
            continue
        
        ### two-frame
        cur_img = cv2.imread(os.path.join('/mnt/frames', video_name, frame_name))     
        prev_img = cv2.imread(os.path.join('/mnt/frames', video_name, prev_frame_name))  
        if  h * w > 320 * 240:
            cur_img = cv2.resize(cur_img, (320, 240)) 
            prev_img = cv2.resize(prev_img, (320, 240)) 
        ##   
        framediff = getFrameDiff(prev_img, cur_img)
        framediff_prec = framediff/ (h * w * 1.0)

        ##
        pilcur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        pilprev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)
        pil_cur = Image.fromarray(pilcur_img)
        pil_prev = Image.fromarray(pilprev_img)
        phash_v = phash(pil_prev, pil_cur) 
        
        ##
        hist_score = colorHistSim(prev_img, cur_img)

        ##
        sprev_img = cv2.resize(prev_img, (160,120))
        scur_img = cv2.resize(cur_img, (160,120))
        sift_score = getSIFTMatchingSim(sprev_img, scur_img)            
        ##
        surf_score = getSURFMatchingSim(sprev_img, scur_img)            
        ###


        x[SOBEL] = cv['sobel'][0]
        x[ILLU] = cv['illu'][0]
        x[FRAME_DIFF] = framediff_prec
        x[PHASH] = phash_v
        x[COLORHIST] = hist_score
        x[SIFTMATCH] = sift_score
        x[SURFMATCH] = surf_score
 
        x = scale_feature(x, range_value) 
        p_label, dummy, dummy = svm_predict([1], [x], svm_model, '-q')
    
        p_label = p_label[0]
        #print p_label
        if int(p_label) == 1:
            selected_fids += [fid]
            print selected_fids
            prev_fid = fid

    pred_trace['picked_f'] = selected_fids
    pred_trace['total_frame'] =  n_frames
    pred_trace['picked_rate'] = len(selected_fids)/(n_frames * 1.0)
    print video_name, selected_fids, '\n' , n_frames, pred_trace['picked_rate']
    
    return pred_trace

def predict_videos(model_fname, model):
    
    #svm_train_0.4_8733_3_0_10.model
    output_trace_foldername =  os.path.join(PRED_TRACELOG, 'trace_' +  model_fname[10:-6])
    if not os.path.exists(output_trace_foldername):
        os.makedirs(output_trace_foldername)
    
    videos = open('/mnt/video_list.txt').read().split()
    for video_name in videos: 
        pred_trace = predict_video(video_name, model)
        print video_name
        with open(os.path.join(output_trace_foldername, video_name + '.pickle'), 'wb') as fh:
            pickle.dump(pred_trace, fh)

def evaluate_trace(video_name, pred_trace):

    OPTIMAL_INPUT_FOLDER = '/home/t-yuche/admission-control/eval/optimal-log/optimal-tf'
    UNIMPORTANTWORD_THRESH = 0.5
    SERVER_STORAGE_FRAMES = 5 * 30 # 5 sec * 30 fps
    SLIDE_SIZE_FRAMES = 1 * 30 # 1 sec * 30 fps
    video_len_f = pred_trace['total_frame']
    _frames = [str(x) + '.jpg' for x in pred_trace['picked_f']]

    with open(os.path.join(OPTIMAL_INPUT_FOLDER, video_name + '_' + str(UNIMPORTANTWORD_THRESH) + '.pickle')) as fh:
        optimal_data = pickle.load(fh)

    rcnn_dict, vgg_dict, dummy, msr_cap_dict, dummy = load_all_modules_dict(video_name)

    start_fids = []
    end_fids = []
    video_start_fid = 0
    video_end_fid = 0
    _scores = []
    while True:

        if video_start_fid > video_len_f - 1:
            break  
        video_end_fid = min(video_start_fid + SERVER_STORAGE_FRAMES, video_len_f)

        key = str(video_start_fid) + '-' + str(video_end_fid)

        ''' Optimal '''
        optimal_tf = optimal_data[key]
        #print optimal_tf

        ''' Our scheme '''
        _range_fids = get_inrange_fids(video_start_fid, video_end_fid, _frames)
        _inrange_frames = [os.path.join('/mnt/frames', video_name, str(x) + '.jpg') for x in _range_fids] 
        _tf_list = combine_all_modeldicts(vgg_dict, msr_cap_dict, rcnn_dict, _inrange_frames)
        _tf = get_combined_tfs(_tf_list)
        _score = detailed_measure(optimal_tf, _tf) 

        #print greedy_tf 
        #print uniform_score, greedy_score

        _scores += [_score]
        video_start_fid +=  SLIDE_SIZE_FRAMES 

    if -1 in _scores:
        _scores.remove(-1) 

    return _scores

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print 'Usage:', sys.argv[0], '1st_model_name 2nd_model_name pvid'
        exit()   
    

    fst_model_name = sys.argv[1]
    snd_model_name = sys.argv[2]
    pvid = int(sys.argv[3])-1
   
    # 1st layer model
    with open(os.path.join(FST_MODELS, fst_model_name)) as fh:
        fst_clf = pickle.load(fh) 

    # 2nd layer model 
    global range_value 
    range_value = load_range_file() 
    snd_model = svm_load_model(os.path.join(SND_MODELS, snd_model_name))


    # output trace and score folder
    output_trace_foldername = os.path.join(PRED_TRACELOG, 'trace_' +  fst_model_name + '_' + snd_model_name[:-6])
    if not os.path.exists(output_trace_foldername):
        os.makedirs(output_trace_foldername)

    output_score_foldername = os.path.join(SCORE_OUTPUT_FOLDER, 'score_' + fst_model_name + '_' + snd_model_name[:-6]) 
    if not os.path.exists(output_score_foldername):
        os.makedirs(output_score_foldername)

    videos = open('/mnt/video_list.txt').read().split()
    for vid, video_name in enumerate(videos):

        print video_name 
        if vid != pvid:
            continue

        output_trace_filepath = os.path.join(output_trace_foldername, video_name + '.pickle') 
        output_score_filepath = os.path.join(output_score_foldername, video_name + '.pickle') 
        #if os.path.exists(output_trace_filepath):
        #    print 'exists'
        #    break
        pred_trace = predict_video(video_name, fst_clf, snd_model)
        with open(output_trace_filepath, 'wb') as fh:
            pickle.dump(pred_trace, fh)

        _scores = evaluate_trace(video_name, pred_trace)
        with open(output_score_filepath, 'wb') as fh:
            pickle.dump(_scores, fh)
        print np.mean(_scores)
        break
