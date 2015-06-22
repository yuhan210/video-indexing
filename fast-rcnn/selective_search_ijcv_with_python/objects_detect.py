#!/usr/bin/env python
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import json
import os

import tempfile
import shlex
import subprocess
import scipy.io

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}

script_dirname = os.path.abspath(os.path.dirname(__file__))

def get_windows(image_fnames, cmd='selective_search'):
    """
    Run MATLAB Selective Search code on the given image filenames to
    generate window proposals.

    Parameters
    ----------
    image_filenames: strings
        Paths to images to run on.
    cmd: string
        selective search function to call:
            - 'selective_search' for a few quick proposals
            - 'selective_seach_rcnn' for R-CNN configuration for more coverage.
    """
    # Form the MATLAB script command that processes images and write to
    # temporary results file.
    f, output_filename = tempfile.mkstemp(suffix='.mat')
    os.close(f)
    fnames_cell = '{' + ','.join("'{}'".format(x) for x in image_fnames) + '}'
    command = "{}({}, '{}')".format(cmd, fnames_cell, output_filename)
    print(command)

    # Execute command in MATLAB.
    mc = "matlab -nojvm -r \"try; {}; catch; exit; end; exit\"".format(command)
    pid = subprocess.Popen(
        shlex.split(mc), stdout=open('/dev/null', 'w'), cwd=script_dirname)
    retcode = pid.wait()
    if retcode != 0:
        raise Exception("Matlab script did not exit successfully!")

    # Read the results and undo Matlab's 1-based indexing.
    all_boxes = list(scipy.io.loadmat(output_filename)['all_boxes'][0])
    subtractor = np.array((1, 1, 0, 0))[np.newaxis, :]
    all_boxes = [boxes - subtractor for boxes in all_boxes]

    # Remove temporary file, and return.
    os.remove(output_filename)
    if len(all_boxes) != len(image_fnames):
        raise Exception("Something went wrong computing the windows!")
    return all_boxes

if __name__ == '__main__':

def Detect(net, image_path):
    
    """Detect object classes in an image assuming the whole image is an object."""
    # Load the image
    im = cv2.imread(image_path)
    h, w, c = im.shape
    
    # TODO: Run selective search first
    object_props = get_windows(image_path)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, np.array([[0, 0, w, h]]))
    scores = scores[0]
    timer.toc()
 
    # get top 6 prediction
    pred_classes = [CLASSES[idx] for idx in ((-scores).argsort()[:6]).tolist()]
    conf = [ (-1) * prob for prob in np.sort(-scores)[:6].tolist()]
    
    img_blob = {}
    img_blob['image_path'] = image_path
    img_blob['pred'] = {'text': pred_classes, 'conf': conf}
    img_blob['rcnn_time'] = timer.total_time

    return img_blob

def parse_args():

    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    parser.add_argument('--video_folder', dest = 'video_folder', type=str, 
                        default='/home/t-yuche/deep-video/data/frames')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
                            'test.prototxt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'fast_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_fast_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()

    video_folder = args.video_folder    
    
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    
    for v in os.listdir(video_folder):
        print v 
        blob = {}
        blob['imgblobs'] = []
       
        frame_folder = os.path.join(video_folder, v)
        
        for f in os.listdir(frame_folder):
            frame_path = os.path.join(frame_folder, f) 
        
            img_blob = Detect(net, frame_path) 

            blob['imgblobs'] += [img_blob]

        json_filename = v + '_multi_rcnnrecog.json'
        json.dump(blob, open(os.path.join('/home/t-yuche/frame-analysis/rcnn-info/', json_filename), 'w'))

