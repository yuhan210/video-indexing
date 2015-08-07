import os
import json
import sys
import argparse
import time
import Image
import numpy as np
import pandas as pd
from scipy.misc import imread, imresize

import cPickle as pickle
TOOL_PATH = '/home/t-yuche/clustering/tools'
sys.path.append(TOOL_PATH)
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--video_name',
                    help='video name')
parser.add_argument("--label_file",
                    default="/home/t-yuche/caffe/data/ilsvrc12/synset_words.txt",
                    help="Index to label file."
                    )

args = parser.parse_args()
caffepath = '/home/t-yuche/caffe/python'
sys.path.append(caffepath)

import caffe

def predict(in_data, net):
    """
    Get the features for a batch of data using network

    Inputs:
    in_data: data batch
    """

    out = net.forward(**{net.inputs[0]: in_data})
    features = out[net.outputs[0]].squeeze(axis=(2,3))
    return features

def classify(in_data, net):

    out = net.forward_all(**{net.inputs[0]: in_data})
    predictions = out[net.outputs[0]] 
    all_labels = []
    all_confs = []
    
    for p in predictions:
        indices = (-p).argsort()[:5]
        label_predictions = labels[indices].tolist()
        all_labels += [label_predictions]
        all_confs += [np.sort(-p)[:5].tolist()]
    return (all_labels, all_confs)


def batch_predict(filenames, net, labels):
    """
    Get the features for all images from filenames using a network

    Inputs:
    filenames: a list of names of image files

    Returns:
    an array of feature vectors for the images in that file
    """
    blob = {}
    blob['imgblobs'] = []
    N, C, H, W = net.blobs[net.inputs[0]].data.shape
    F = net.blobs[net.outputs[0]].data.shape[1]
    Nf = len(filenames)
    #allftrs = np.zeros((Nf, F))
    allpreds = []
    for i in range(0, Nf, N):
        tic = time.time()
        in_data = np.zeros((N, C, H, W), dtype=np.float32)

        batch_range = range(i, min(i+N, Nf))
        batch_filenames = [filenames[j] for j in batch_range]
        Nb = len(batch_range)

        batch_images = np.zeros((Nb, 3, H, W))
        for j,fname in enumerate(batch_filenames):
            im = np.array(Image.open(fname))
            if len(im.shape) == 2:
                im = np.tile(im[:,:,np.newaxis], (1,1,3))
            # RGB -> BGR
            im = im[:,:,(2,1,0)]
            # mean subtraction
            im = im - np.array([103.939, 116.779, 123.68])
            # resize
            im = imresize(im, (H, W))
            # get channel in correct dimension
            im = np.transpose(im, (2, 0, 1))
            batch_images[j,:,:,:] = im

        # insert into correct place
        in_data[0:len(batch_range), :, :, :] = batch_images

        (ps,cs) = classify(in_data, net)
        toc = time.time()
        print 'time: ', str(toc-tic), 'per image: ', (toc-tic)/len(batch_range)
        #allpreds += [ps]
        for j, fname in enumerate(batch_filenames):
            img_blob = {}
            img_blob['img_path'] = fname
            img_blob['pred'] = {'text': ps[j], 'conf': cs[j]} 
            blob['imgblobs'].append(img_blob) 
        
        # predict features
        #ftrs = predict(in_data, net)
        #for j in range(len(batch_range)):
        #    allftrs[i+j,:] = ftrs[j,:]
        print 'Done %d/%d files' % (i+len(batch_range), len(filenames))

    return blob

def loadKeyFrames(video_name):

    KEYFRAME_FOLDER = '/home/t-yuche/gt-labeling/frame-subsample/keyframe-info'
    keyframe_file = os.path.join(KEYFRAME_FOLDER, video_name + '_uniform.json')

    with open(keyframe_file) as json_file:
        keyframes = json.load(json_file)

    return keyframes


'''
if args.cpu:
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()
'''
caffe.set_mode_gpu()
model_def = '/home/t-yuche/neuraltalk/python_features/VGG_ILSVRC_16_layers_deploy.prototxt' 
model = '/home/t-yuche/caffe/models/vgg_ilsvrc_16/VGG_ILSVRC_16_layers.caffemodel'
net = caffe.Net(model_def, model, 0)
video_name = args.video_name

# set up output file
OUTPUT_FOLDER = '/mnt/tags/vgg-classify-keyframe' 
output_file = os.path.join(OUTPUT_FOLDER, video_name + '_recog.json')

# load key frames
FRAME_FOLDER = '/mnt/frames'
keyframes = loadKeyFrames(video_name)
keyframe_filenames = [ os.path.join(FRAME_FOLDER, video_name, x['key_frame']) for x in keyframes['img_blobs'] ]
 

# Load label file
with open(args.label_file) as f:
    labels_df = pd.DataFrame([
        {
            'synset_id':l.strip().split(' ')[0],
            'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
        }
        for l in f.readlines()
    ])
labels = labels_df.sort('synset_id')['name'].values

# Load processed frames 
recog_data = load_video_recog('/mnt/tags/vgg-classify', video_name)
processed_frames = [x['img_path'] for x in recog_data]

# Prnue filenames
filenames = []
for f in keyframe_filenames:
    if f not in processed_frames:
        filenames += [f] 


allpreds = batch_predict(filenames, net, labels)

# Combine old detection and the current one
allpreds['imgblobs'] = allpreds['imgblobs'] + recog_data 

# dump result struct to file
print 'writing predictions to %s...' % (output_file, )
json.dump(allpreds, open(output_file, 'w'))
