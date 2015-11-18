import os
import sys
import pickle
import random
sys.path.append('/home/t-yuche/mcdnn/caffe/python')
import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format
import numpy as np
import plyvel
import shutil

videos = open('/mnt/video_list.txt').read().split()
GREEDYTHRESH = 0.8
db_name = "_train"


def set_mean(mean, img_dims):
    mean = caffe.io.resize_image(mean.transpose((1,2,0)),
                                         img_dims).transpose((2,0,1))

    return mean

def load_preprocess_config():
    model_def = './models/caffe_alex.prototxt'
    mean_file = '/home/t-yuche/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'

    raw_scale = 255
    channel_order = (2,1,0) 
    img_dims = (227, 227)
    crop_dims = np.array((227, 227))

    mean = np.load(mean_file)
    mean = set_mean(mean, img_dims)

    net = {'mean':mean, 'raw_scale': raw_scale, 'channel_order': channel_order, 'image_dims': img_dims, 'crop_dims': crop_dims}
    return net

def prep_input_original_img(net, prev_img, cur_img):

    input_ = np.zeros((1, prev_img.shape[2] + cur_img.shape[2], 
        net['image_dims'][0], net['image_dims'][1]),
        dtype=np.float32)

    # resize
    prev_img = caffe.io.resize_image(prev_img, net['image_dims'])
    cur_img = caffe.io.resize_image(cur_img, net['image_dims'])

    

    #### substract channel mean
    prev_img = prev_img[:, :, net['channel_order']] 
    cur_img = cur_img[:, :, net['channel_order']]

    # h, w, k to k, h, w
    prev_img = prev_img.transpose((2,0,1))
    cur_img = cur_img.transpose((2,0,1))
    
    #print prev_img.shape, 'should be (3, 227, 227)'
    #print prev_img

    # * raw_scale 
    prev_img *= net['raw_scale']
    cur_img *= net['raw_scale'] # k , h, w
 
    # substract from mean
    prev_img -= net['mean']
    cur_img -= net['mean']
    ###
     
    input_[0] = np.concatenate((cur_img, prev_img), axis = 0)

    # Take center crop.
    center = np.array(net['image_dims']) / 2.0 
    crop = np.tile(center, (1, 2))[0] + np.concatenate([
        -net['crop_dims'] / 2.0,
        net['crop_dims'] / 2.0 
    ])  
    input_ = input_[:, :, crop[0]:crop[2], crop[1]:crop[3]]

    return input_


def generate_db(db_name, train_data, create_if_exists=False):
    net_config = load_preprocess_config()
    if os.path.exists(db_name):
        if create_if_exists:
            shutil.rmtree(db_name)
        else:
            return

    db = plyvel.DB(db_name, create_if_missing=True, error_if_exists=True)
    wb = db.write_batch()

    count = 0
    cur_vid = 0  
    for vid, video_name in enumerate(train_data):
        samples = train_data[video_name] 
        for sample in samples:

            prev_img_path = os.path.join('/mnt/frames', video_name, str(sample[0]) + '.jpg' )
            curr_img_path = os.path.join('/mnt/frames', video_name, str(sample[1]) + '.jpg' )
            if not os.path.exists(prev_img_path) or not os.path.exists(curr_img_path):
                print 'Warning:', video_name, sample, 'does not exist'
                continue
            prev_img = caffe.io.load_image(prev_img_path)
            curr_img = caffe.io.load_image(curr_img_path)

            
            data_ = prep_input_original_img(net_config, prev_img, curr_img)
            datum = caffe.io.array_to_datum(np.array(data_[0, :, :, :]).astype(float), sample[2])
            wb.put('%08d' % count, datum.SerializeToString())
            count += 1

            cur_vid = vid 
            if count % 1000 == 0:
                print count, '/' , len([train_data[x] for x in train_data])
                wb.write()
                wb = db.write_batch()
    print cur_vid, count
    wb.write()
    db.close()

def generate_traindata():

    GREEDYFOLDER = '/home/t-yuche/admission-control/greedy/window-greedy-log-0.5'

    train_data = {}
    for video_name in videos:
        
        # read greedy trace
        greedypath = os.path.join(GREEDYFOLDER, video_name + '_' + str(GREEDYTHRESH)  + '_gtframe.pickle')
        gt_data = pickle.load(open(greedypath))
        gt_picked_fid = gt_data['picked_f']
        total_frame_n = gt_data['total_frame'] 

        train_pairs = []
        prev_fid = 0
        for fid in xrange(1, total_frame_n):
        
            if fid in gt_picked_fid: # positive
                train_pairs += [(prev_fid, fid, 1)]
                prev_fid = fid
            else: # negative
                train_pairs += [(prev_fid, fid, 0)] 

        train_data[video_name] = train_pairs

    return train_data


def train_cnn(db_name, train_data):

    caffe.set_mode_gpu()
    if train_data == None:
        n_samples = 1000
    else:
        n_samples = len(train_data) 
    
    solver_param = caffe_pb2.SolverParameter()
    with open('./models/solver_template.prototxt') as f:
        google.protobuf.text_format.Merge(f.read(), solver_param)


    # nitem 
    solver_param.stepsize = n_samples
    solver_param.max_iter = int(n_samples * 10) 

    print solver_param 
    with open('./models/solver_template_a7_' + db_name + '.prototxt', 'w') as f:
        f.write(google.protobuf.text_format.MessageToString(solver_param))
    name = f.name

    print name
    solver = caffe.SGDSolver(name)

    solver.solve()

    trained_model = str(solver_param.snapshot_prefix) + '_iter_' + str(solver_param.max_iter) + '.caffemodel'

    return trained_model



if __name__ == "__main__":

    print 'Generate train_data'
    all_pairs = generate_traindata()
    train_data = {}
    for video_name in all_pairs:
        
        pos_samples = filter(lambda x: x[2] == 1, all_pairs[video_name])
        neg_samples = filter(lambda x: x[2] == 0, all_pairs[video_name])
        n_pos_samples = len(pos_samples)

         
        # 1. for each video select half of the pos sample n_hpos, and m_hpos/2 neg samples
        train_pos_samples = random.sample(pos_samples, n_pos_samples/2)
        train_neg_samples = random.sample(neg_samples, n_pos_samples/4) 
        train_data[video_name] = train_pos_samples + train_neg_samples 
    
    print 'Generate cnn database' 
    generate_db(db_name, train_data, True)
    '''
    print 'Start training'
    train_cnn(db_name, None)   
    #train_cnn(db_name, train_data)   
    ''' 
    
