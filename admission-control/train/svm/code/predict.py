from utils import *
from vision import *
import pickle
import sys
sys.path.append('../liblinear/python')
from svmutil import *
import sys

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

PRED_TRACELOG = './pred-trace'
MV_INPUT_FOLDER = '/home/t-yuche/admission-control/train/mv_log'

def load_range_file(scale_file):
   
    scale_value = {}
    with open(scale_file) as f:
        for line in f.readlines():
            line = line.strip()
            segs = line.split(' ') 
            if line.split(' ') == 3:
                idx = int(segs[0])
                min_v = int(segs[1])
                max_v = int(segs[2])
                scale_value[idx] = (min_v, max_v)

    return scale_value

def scale_feature(sample):
    upper = 1.0
    lower = -1.0

    for fid in sample:
        value = sample[fid]
        if fid in range_value.key():
            if value <= scale_value[0]:
                value = lower
            elif value >= scale_value[1]:
                value = upper
            else:
                value = lower + (upper - lower) * (value - scale_value[0])/(scale_value[1] - scale_value[0])
        sample[fid] = value

    return sample

def predict_video(video_name, svm_model):
   
    n_frames = get_video_frame_num(video_name) 
    pred_trace = {'picked_f': [], 'total_frame': 0,'picked_rate': 0.0}
 
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
        prev_frame_name = str(prev_fid) + 'jpg' 
        # generate features
        #### single frame features
        enc = encdata[frame_name]
        if enc['type'] == 'P': 
            x[ENC_TYPE] = 0
        else:
            x[ENC_TYPE] = 1

        cv = cvdata[frame_name]
        mv = mv_features[fid]
        ###
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

        x[IMG_WIDTH] = w
        x[IMG_HEIGHT] = h
        x[ENC_SIZE] = enc['size']
        x[MV_SIZE] = mv[0] 
        x[MV_MEAN] = mv[fid][1]
        x[MV_MAX] = mv[fid][2]
        x[MV_MIN] = mv[fid][3]
        x[SOBEL] = cv['sobel'][0]
        x[ILLU] = cv['illu'][0]
        x[FRAME_DIFF] = framediff_prec
        x[PHASH] = phash_v
        x[COLORHIST] = hist_scor
        x[SIFTMATCH] = sift_score
        x[SURFMATCH] = surf_score
  
 
        # predict if we should pick this frame 
        # scale feature
        x = scale_feature(x) 
         
        # predict
        p_label, dummy, dummy = svm_predict([1], x, svm_model)

        if int(p_label) == 1:
            selected_fids += [fid]
            prev_fid = fid

    return selected_fids 

if __name__ == "__main__":

    videos = open('/mnt/video_list.txt').read().split()
    model_name = './models/svm_train.model'
    model = svm_load_model(model_name)

    if len(sys.argv) != 2:
        print 'Usage:', sys.argv[0], 'range_file'
        exit()

    range_file = sys.argv[1]
   
    global range_value 
    range_value = load_range_file(range_file) 
 
  
    for video_name in videos: 
        pred_trace = predict_video(video_name, model) 
        #TODO: store predicted trace 
