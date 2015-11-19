import sys
sys.path.append('../liblinear/python')
from svmutil import *
import os

DATA_FOLDER = 'data'
MODEL_FOLDER = 'models'

def train_model(train_data_path, model_output_path, PARAM):

    if os.path.exists(model_output_path):
        return
    y, x = svm_read_problem(train_data_path)
    m = svm_train(y, x, '-t ' + PARAM['-t'] + ' -c ' + PARAM['-c']) 


    svm_save_model(model_output_path, m)

if __name__ == "__main__":

    PARAM = {'-t': '0',  '-c': '10'}
    
    for train_data_path in os.listdir(DATA_FOLDER):
        #svm_train_0.4_8733_2.data
        print 'Training ', train_data_path
        #segs = train_data_path.split('_')
        #PERCENT_OF_POS_SAMPLES = segs[2]
        #N_POS_SAMPLES = segs[3]
        #P_TO_N_RATIO = segs[4]
        
        model_output_path = os.path.join(MODEL_FOLDER, train_data_path[:-5] + '_' + PARAM['-t'] + '_' + PARAM['-c'] + '.model')
         
 
        train_model(os.path.join(DATA_FOLDER, train_data_path), model_output_path, PARAM)

