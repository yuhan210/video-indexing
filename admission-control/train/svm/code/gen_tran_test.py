# train various training/testing partition
from svm_utils import *
import random
import os

    
def scale_samples(samples):

    features = []
    for sample in samples:
       
        feature = convert_sample_to_dict(sample) 
        feature_scaled = scale_feature(feature, scale_value)
        features += [feature_scaled]
    
    # features = {(0):label, 1:feature_1, 2:feature_2, 3: feature_3...}
    return features



def write_samples(pos_features, neg_features, PERCENT_OF_POS_SAMPLES, N_POS_SAMPLES, N_NEG_SAMPLES, folder = 'data'):
    file_name = os.path.join(folder, 'svm_train_' + str(PERCENT_OF_POS_SAMPLES)  + '_' + str(N_POS_SAMPLES) + '_' + str(N_NEG_SAMPLES) + '.data')

    with open(file_name, 'w') as fh:
        for feature in pos_features:
            output_str = feature_tostring(feature)
            fh.write(output_str + '\n')
    
        fh.flush()
        for feature in neg_features:
            output_str = feature_tostring(feature)
            fh.write(output_str + '\n')

        fh.flush()
        fh.close()
    
if __name__ == "__main__":

    
    samples, pos_count, neg_count = load_train_data('svm_train_all.data')  
    global scale_value
    scale_value = load_range_file() 
     
    # partition all training data into training and testing   
    pos_samples = filter(lambda x:x[0] == 1, samples)
    neg_samples = filter(lambda x:x[0] == 0, samples)

    pos_features = scale_samples(pos_samples) # a list of dict 
    neg_features = scale_samples(neg_samples)
    
    # print len(pos_features), len(neg_features)   
    # 21833 4719259 

    for p in [0.4, 0.5, 0.6, 0.7]:
        print p
        ############## modify!!!
        PERCENT_OF_POS_SAMPLES = p
        N_POS_SAMPLES = int(len(pos_features) * PERCENT_OF_POS_SAMPLES)
        P_TO_N_RATIO = 2
        N_NEG_SAMPLES = int(N_POS_SAMPLES/P_TO_N_RATIO)
        ###############

        sub_pos_features = random.sample(pos_features, N_POS_SAMPLES)
        sub_neg_features = random.sample(neg_features, N_NEG_SAMPLES)
        write_samples(sub_pos_features, sub_neg_features, PERCENT_OF_POS_SAMPLES, N_POS_SAMPLES, N_NEG_SAMPLES)
     
