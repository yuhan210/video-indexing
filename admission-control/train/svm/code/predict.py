from svm_utils import *
from vision import *
import pickle
import sys
sys.path.append('../liblinear/python')
from svmutil import *
import sys
from utils import *

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
        x[MV_MEAN] = mv[1]
        x[MV_MAX] = mv[2]
        x[MV_MIN] = mv[3]
        x[SOBEL] = cv['sobel'][0]
        x[ILLU] = cv['illu'][0]
        x[FRAME_DIFF] = framediff_prec
        x[PHASH] = phash_v
        x[COLORHIST] = hist_score
        x[SIFTMATCH] = sift_score
        x[SURFMATCH] = surf_score
        x[DIST_FROM_PFID] = fid - prev_fid 
 
        # predict if we should pick this frame 
        # scale feature
        # print x
        x = scale_feature(x, range_value) 

        print video_name, fid, x 
        # predict
        p_label, dummy, dummy = svm_predict([1], [x], svm_model)
        
        p_label = p_label[0]
        print p_label
        if int(p_label) == 1:
            selected_fids += [fid]
            prev_fid = fid

    return selected_fids 

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print 'Usage:', sys.argv[0], 'model_path'
        exit()   
 
    model_name = sys.argv[1]

    model = svm_load_model(model_name)
    if model_name[-1] == '/':
        model_name = model_name[0:-1]
    model_fname = model_name.split('/')[-1]

    #svm_train_0.4_8733_3_0_10.model
    output_trace_foldername=  os.path.join(PRED_TRACELOG, 'trace_' +  model_fname[10:-6])
    if not os.path.exists(output_trace_foldername):
        os.makedirs(output_trace_foldername)
 
    global range_value 
    range_value = load_range_file() 
 
  
    videos = open('/mnt/video_list.txt').read().split()
    for video_name in videos: 
        pred_trace = predict_video(video_name, model)
        print pred_trace 
        #TODO: store predicted trace 
        with open(os.path.join(output_trace_foldername, video_name + '.pickle'), 'wb') as fh:
            fh.dump(pred_trace, fh)
        exit() 

