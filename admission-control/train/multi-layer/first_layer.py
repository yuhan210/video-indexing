import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import train_test_split
from lib import *
import time
from sklearn import tree
from sklearn.metrics import accuracy_score
import pickle

FEATURE_NAMES = ['enctype', 'width', 'height', 'encsize', 'mvsize', 'meanmv', 'maxmv', 'minmv', 'dist_from_prevfid']

def compute_accuracy(y_pred, Y):
    assert(len(y_pred) == len(Y))
    correct_count = 0
    conf = [[0,0], [0,0]] 
    for sid, py in enumerate(list(y_pred)):
        conf[Y[sid]][int(py)] += 1 
        if int(py) == int(Y[sid]): 
            correct_count += 1

    print 'Accuracy:', correct_count/(len(y_pred) * 1.0)
    print conf
    print 'Conf metric'
    print conf[0][0]/(sum(conf[0]) * 1.0), conf[0][1]/(sum(conf[0]) * 1.0)
    print conf[1][0]/(sum(conf[1]) * 1.0), conf[1][1]/(sum(conf[1]) * 1.0)



if __name__ == "__main__":

    train_data = load_all_data_first_layer() 
    videos = train_data.keys()

    pos_samples = []   
    neg_samples = []
    
    pos_train_samples = []
    neg_train_samples = []
    pos_test_samples = []
    neg_test_samples = []
    
    all_train_X = []
    all_train_y = []
    all_test_X = []
    all_test_y = []
    dup = 5
    for vid, video_name in enumerate(videos):
        #n_positive_sample = len(train_data[video_name]['1'])

        #X_pos_train, X_pos_test, Y_pos_train, Y_pos_test = train_test_split(train_data[video_name]['1'], [1] * len(train_data[video_name]['1']), train_size = 0.99, random_state = 0)
        pos_samples += train_data[video_name]['1']

        #for x in xrange(dup):
        #    all_train_X += X_pos_train
        #    all_train_y += [1] * len(X_pos_train)

        #X_neg_train, X_neg_test, Y_neg_train, Y_neg_test = train_test_split(train_data[video_name]['0'], [0] * len(train_data[video_name]['0']), train_size = len(X_pos_train) * dup, random_state = 0)

        
        #X_neg_train, X_neg_test, Y_neg_train, Y_neg_test = train_test_split(train_data[video_name]['0'], [0] * len(train_data[video_name]['0']), test_size = (1-TRAINING_SIZE), random_state = 0)
        neg_samples += train_data[video_name]['0']

        '''
        all_train_X += X_neg_train
        all_train_y += [0] * len(X_neg_train)   

        all_test_X += X_pos_test
        all_test_y += [1] * len(X_pos_test)
        all_test_X += X_neg_test
        all_test_y += [0] * len(X_neg_test)
        '''

    #pn_ratio = len(neg_samples)/(len(pos_samples) * 1.0) + 2
    X = pos_samples + neg_samples
    #w = [pn_ratio] * len(pos_samples) + [1] * len(neg_samples)
    Y = [1] * len(pos_samples) + [0] * len(neg_samples)
    
    for x in [0.5, 0.7, 0.9]:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = x, random_state = 0)
        print x, len(X_train)    

        #clf = tree.DecisionTreeClassifier(class_weight={1: pn_ratio, 0: 1})
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train,Y_train)
        outputfilename = 'dct_' + str(len(X_train)) + '_'  + str(x) + '_eqw' 
        with open(outputfilename, 'w') as fh:
            pickle.dump(clf, fh)
        #pred_y = clf.predict(X_test)
        #print accuracy_score(Y_test, pred_y), '\n' 
        #compute_accuracy(pred_y, Y_test)


    #clf = clf.fit(all_train_X, all_train_y)
    #pred_y = clf.predict(all_test_X)
    #compute_accuracy(pred_y, all_test_y)
    #print accuracy_score(pred_y, all_test_y), '\n'


    #print ''
        bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                             algorithm="SAMME",
                             n_estimators=200)
        bdt.fit(X_train, Y_train)
        outputfilename = 'adb_' + str(len(X_train)) + '_' + str(x) + '_eqw' 
        with open(outputfilename, 'w') as fh:
            pickle.dump(bdt, fh)
        #pred_y = bdt.predict(all_test_X)
        #compute_accuracy(pred_y, all_test_y)
