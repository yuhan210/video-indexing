import numpy as np
from sklearn.tree import DecisionTreeClassifier
from operator import itemgetter
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import train_test_split
from lib import *
from sklearn import svm
from sklearn.cross_validation import KFold
import time
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import random
import pickle
import sys
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import  cross_val_score
import matplotlib
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
import pydot
from IPython.display import Image

FEATURE_NAMES = ['enctype', 'width', 'height', 'encsize', 'mvsize', 'meanmv', 'maxmv', 'minmv', 'dist_from_prevfid']
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
OPTIMAL_INPUT_FOLDER = '/home/t-yuche/admission-control/eval/optimal-log/optimal-tf'

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

def report(grid_scores, n_top=3):
    """Report top n_top parameters settings, default n_top=3.

    Args
    ----
    grid_scores -- output from grid or random search
    n_top -- how many to report, of top models

    Returns
    -------
    top_params -- [dict] top parameter settings found in
                  search
    """
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
               score.mean_validation_score,
               np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

    return top_scores[0].parameters




def run_gridsearch(X, y, clf, param_grid, cv=5):
    """Run a grid search for best Decision Tree parameters.

    Args
    ----
    X -- features
    y -- targets (classes)
    cf -- scikit-learn Decision Tree
    param_grid -- [dict] parameter settings to test
    cv -- fold of cross-validation, default 5

    Returns
    -------
    top_params -- [dict] from report()
    """
    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,
                               cv=cv)
    start = time.time()
    grid_search.fit(X, y)

    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time.time() - start,
                len(grid_search.grid_scores_)))

    top_params = report(grid_search.grid_scores_, 10)
    return  top_params


def compute_accuracy(y_pred, Y):
    assert(len(y_pred) == len(Y))
    correct_count = 0
    conf = [[0,0], [0,0]] 
    for sid, py in enumerate(list(y_pred)):
        conf[Y[sid]][int(py)] += 1 
        if int(py) == int(Y[sid]): 
            correct_count += 1

    print 'Recall:', correct_count/(len(Y) * 1.0)
    print 'Precision:', correct_count/(len(y_pred) * 1.0)
    print conf[0][0], conf[0][1]
    print conf[1][0], conf[1][1]
    print 'Conf metric'
    print conf[0][0]/(sum(conf[0]) * 1.0), conf[0][1]/(sum(conf[0]) * 1.0)
    print conf[1][0]/(sum(conf[1]) * 1.0), conf[1][1]/(sum(conf[1]) * 1.0)

def predict_video(video_name, enc_model, clf_model):
   
    n_frames = get_video_frame_num(video_name) 
    pred_trace = {'picked_f': [], 'total_frame': 0, 'picked_rate': 0.0, 'proc_stage': []}
 
    encdata = getMetadata(video_name) 
    cvdata = getCVInfoFromLog(video_name)
    mv_file = os.path.join(MV_INPUT_FOLDER, video_name + '.pickle')
    with open(mv_file) as fh:
        mv_features = pickle.load(fh)
    w = encdata['metadata']['w'] 
    h = encdata['metadata']['h']

    selected_fids = [0]
    prev_fid = 0
    stages = [-1]
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
            stages += [0]
            continue
        
        ### two-frame
        cur_img = cv2.imread(os.path.join('/home/t-yuche/frames', video_name, frame_name))     
        prev_img = cv2.imread(os.path.join('/home/t-yuche/frames', video_name, prev_frame_name))  
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
 
        #x = scale_feature(x, range_value) 
        #p_label, dummy, dummy = svm_predict([1], [x], svm_model, '-q')
        snd_layer_f = [x[ENC_TYPE], x[IMG_WIDTH], x[IMG_HEIGHT], x[ENC_SIZE], x[MV_SIZE], x[MV_MEAN], x[MV_MAX], x[MV_MIN], x[SOBEL], x[ILLU], x[FRAME_DIFF], x[PHASH], x[COLORHIST], x[SIFTMATCH], x[SURFMATCH], x[DIST_FROM_PFID]]
        pred_y = clf_model.predict(snd_layer_f) 
         
        #print p_label
        if int(pred_y) == 1:
            stages += [2]
            selected_fids += [fid]
            prev_fid = fid
        else:
            stages += [1]

    pred_trace['picked_f'] = selected_fids
    pred_trace['total_frame'] =  n_frames
    pred_trace['picked_rate'] = len(selected_fids)/(n_frames * 1.0)
    pred_trace['proc_stage'] = stages
    #print video_name, selected_fids, '\n' , n_frames, pred_trace['picked_rate']
    
    return pred_trace


if __name__ == "__main__":

    thresh = sys.argv[1]
    #pvid = int(sys.argv[2])-1
    train_data = load_all_data_second_layer('/home/t-yuche/admission-control/train/train_' + str(thresh) +'_out.data') 
    #snd_train_data = load_all_data_second_layer('/home/t-yuche/admission-control/train/train_' + str(thresh) +'_out.data')
    videos = train_data.keys()
    #fh = open('./log', 'w')
    pos_samples = []   
    neg_samples = []

    snd_pos_samples = []
    snd_neg_samples = []    

    videos = open('/home/t-yuche/video_list.txt').read().split()
    # divide videos into 6 chunks
    kf = KFold(len(videos), 6) 
    for train_idx, test_idx in kf:
 
        test_pos_samples = []
        test_neg_samples = []
        pos_samples = []   
        neg_samples = []

        test_video_names = [video_name for idx, video_name in enumerate(videos) if idx in test_idx]   
        #train_video_names = [video_name for idx, video_name in enumerate(videos) if idx in train_idx]  

        for tid, video_name in enumerate(videos):
            if video_name in test_video_names:
                # load test data
                continue
                #test_pos_samples += [x[:-1] for x in train_data[video_name]['1']]
                #test_neg_samples += [x[:-1] for x in train_data[video_name]['0']]
           
            else: 
                pos_samples += [[x[1]] for x in train_data[video_name]['1']]
                neg_samples += [[x[1]] for x in train_data[video_name]['0']]
            
            #snd_pos_samples += snd_train_data[train_video_name]['1']
            #snd_neg_samples += snd_train_data[train_video_name]['0']

        X_pos_train = pos_samples
        #X_neg_train = neg_samples
        X_neg_train, dummy, dummy, dummy = train_test_split(neg_samples, [0] * len(neg_samples), train_size = len(X_pos_train), random_state = 0)
    
         
        print len(X_pos_train), len(X_neg_train)
         
        #snd_X_pos_train = snd_pos_samples
        #snd_X_neg_train = snd_neg_samples
        #snd_X_neg_train, dummy, dummy, dummy = train_test_split(snd_neg_samples, [0]* len(snd_neg_samples),train_size = int(len(snd_neg_samples) * 0.8), random_state = 0)

        X_train = X_neg_train + X_pos_train
        Y_train = [0] * len(X_neg_train) + [1] * len(X_pos_train)

        #fst_clf = tree.DecisionTreeClassifier(random_state = 0, class_weight={1: int(len(X_neg_train)/(len(X_pos_train) * 1.0))*2})
        fst_clf = tree.DecisionTreeClassifier(random_state = 0,max_depth = 2 )
        fst_clf = fst_clf.fit(X_train,Y_train)
        dot_data = StringIO()
        tree.export_graphviz(fst_clf, filled = True, out_file=dot_data) 
        graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
        graph.write_pdf("tree.pdf") 
       
        for video_name in test_video_names:
            print video_name
            selected_fids = [x[-1] for x in train_data[video_name]['1']]
             
            p_samples = train_data[video_name]['1']
            n_samples = train_data[video_name]['0']
            samples = p_samples + n_samples
            sorted(samples, key = itemgetter(-1))
            scatter = []
            
            #for sfid in selected_fids:
                
            for s in samples:
                y_pred = fst_clf.predict([s[1]])
                cfid = s[-1]
                scatter += [(cfid, y_pred)]
            
            plt.scatter([x[0] for x in scatter], [x[1] for x in scatter])
            plt.scatter(selected_fids, [3 for x in selected_fids])
            plt.show()
            break
        #X_test = test_pos_samples + test_neg_samples
        #Y_test = [1] * len(test_pos_samples) + [0] * len(test_neg_samples)
        #Y_pred = fst_clf.predict(X_test)
        #compute_accuracy(Y_pred, Y_test)
        #print confusion_matrix(Y_test, Y_pred)
        #clf = svm.SVC()
        #clf = svm.SVC(kernel='linear', class_weight={1: int(len(X_neg_train)/(len(X_pos_train) * 1.0))})
        #fst_clf = clf.fit(X_train, Y_train)
        '''
        conf = [[0,0], [0,0]] 
        check_range = 0
        for video_name in test_video_names:
            # get selected fids 
            selected_fids = [x[-1] for x in train_data[video_name]['1']]
            
            for sample in train_data[video_name]['1']:
                fid = sample[-1]
                Y_pred = fst_clf.predict(sample[:-1])
                
                if Y_pred == 1:
                    conf[1][1] += 1
                else:
                    conf[1][0] += 1
 
            for sample in train_data[video_name]['0']:
                fid = sample[-1]
                Y_pred = fst_clf.predict(sample[:-1])
                
                if Y_pred == 1:
                    conf[0][1] += 1
                else:
                    conf[0][0] += 1
                in_range = False
                for sfid in selected_fids:                
                    if fid >= sfid - check_range and fid <= sfid + check_range:
                        in_range = True
                        break 
                if in_range:
                    if Y_pred == 1:
                        conf[1][1] += 1
                    if Y_pred == 0:
                        conf[0][0] += 1
                else:
                    if Y_pred == 0:
                        conf[0][0] += 1
                    else:     
                        conf[0][1] += 1
        '''
 
        #print '--------------'
        #print conf
        #print conf[0][0]/(sum(conf[0]) * 1.0), conf[0][1]/(sum(conf[0]) * 1.0)
        #print conf[1][0]/(sum(conf[1]) * 1.0), conf[1][1]/(sum(conf[1]) * 1.0)
        #print test_video_name
        #compute_accuracy(Y_pred, Y_test)
        #print confusion_matrix(Y_test, Y_pred)

        #X_train = snd_X_neg_train + snd_X_pos_train
        #Y_train = [0] * len(snd_X_neg_train) + [1] * len(snd_X_pos_train)
        #snd_clf = tree.DecisionTreeClassifier(random_state = 0)
        #snd_clf = snd_clf.fit(X_train,Y_train)

                  
        #break 
