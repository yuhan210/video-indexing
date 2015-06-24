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

import time
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

def get_windows(image_fnames, output_files, cmd='selective_search_rcnn'):
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

    # Form the MATLAB script command that processes images and write to results file.
    fnames_cell = '{' + ','.join("'{}'".format(x) for x in image_fnames) + '}'
    command = "{}({}, '{}')".format(cmd, fnames_cell, output_filename)
    #print(command)

    # Execute command in MATLAB.
    mc = "matlab -nojvm -r \"try; {}; catch; exit; end; exit\"".format(command)
    tic = time.time()
    pid = subprocess.Popen(
        shlex.split(mc), stdout=open('/dev/null', 'w'), cwd=script_dirname)
    retcode = pid.wait()
    toc = time.time()
    if retcode != 0:
        raise Exception("Matlab script did not exit successfully!")

    # Read the results and undo Matlab's 1-based indexing.
    all_boxes = list(scipy.io.loadmat(output_filename)['all_boxes'][0])
    subtractor = np.array((1, 1, 0, 0))[np.newaxis, :]
    all_boxes = [boxes - subtractor for boxes in all_boxes]

    if len(all_boxes) != len(image_fnames):
        raise Exception("Something went wrong computing the windows!")
    
    return (toc-tic), all_boxes


def Detect(net, image_path, object_proposals):
    
    """Detect object classes in an image assuming the whole image is an object."""
    # Load the image
    im = cv2.imread(image_path)
    h, w, c = im.shape

    # Detect all object classes and regress object bounds
    tic = time.time()
    scores, boxes = im_detect(net, im, object_proposals)
    toc = time.time()
    detect_time = (toc-tic)
    # Visualize detections for each class
    '''
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls in classes:
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls,
                                                                    CONF_THRESH)
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
    '''
    # need to process each proposal
    img_blob = {}
    img_blob['img_path'] = image_path
    img_blob['detections'] = []
  
    # sort for each row
    sort_idxs = np.argsort(-scores, axis = 1).tolist()

    # for each proposal
    for idx, idx_rank in enumerate(sort_idxs):
        
        # get top-6
        t_boxes = []
        preds = []
        confs = [] 
        idx_rank = idx_rank[:6] # a list
        # for top-6 class
        for cls_ind in idx_rank:
            t_boxes += [ boxes[idx, 4*cls_ind:4*(cls_ind+1)].tolist() ]
            preds += [ CLASSES[cls_ind] ]
            confs += [ scores[idx, cls_ind] ] 
   
        img_blob['detections']  += [[t_boxes, preds, confs]]
    

    return detect_time, img_blob

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

    ## output
    ## proposals.json, rcnn_output.json (blob), selective_search_time.json (bbox_time), digesting_proposals.json (detect_time)
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

    batch_size = 30

    # for each video
    for v in os.listdir(video_folder):
        print v
        blob = {}
        blob['img_blobs'] = []
        
        regions = {}
        regions['img_blobs'] = []

        bbox_time_log = []

        detect_time_log = {}
        detect_time_log['img_blobs'] = []
        
        frame_folder = os.path.join(video_folder, v)
        image_filenames = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder)]
        image_filenames = sorted(image_filenames, key=lambda x: int(x.split('/')[-1].split('.')[0]))  

        batch_count = 0 
        for i in range(0, len(image_filenames), batch_size):
            batch_range = range(i, min(i+batch_size, len(image_filenames)))
            batch_filenames = [image_filenames[j] for j in batch_range]
            
            output_filename = '/tmp/' + v + '_batch_' + str(batch_count) + '.mat'         
            
            # return a list of arrays 
            selective_search_time, all_boxes = get_windows(batch_filenames, output_filename)
            
            ## bbox generation time
            bbox_time_log += [{'time':selective_search_time , 'n_frames': len(batch_filenames)}]
        
            # cannot do in batches, iterate through all the images
            # for each image
            for idx, img_name in enumerate(batch_filenames):
                proposals = all_boxes[idx]
                detect_time, img_blob = Detect(net, img_name, proposals)
                blob['img_blobs'] += [img_blob]

                region = {}
                region['img_path'] = img_name
                region['proposals'] = proposals.tolist()
                regions['img_blobs'] += [region]
                    
                detect_log = {}
                detect_log['img_name'] = img_name
                detect_log['detect_time'] = detect_time
                detect_time_log['img_blobs'] += [detect_log]
                
            batch_count += 1
  
      
        result_folder = '/home/t-yuche/frame-analysis/rcnn-mulreg-info'
        # write execution time
        proposal_t_json_filename = v + '_proposal_time.json'
        detect_t_json_filename = v + '_detect_time.json'
        
        json.dump(bbox_time_log, open(os.path.join(result_folder, proposal_t_json_filename), 'w'))
        json.dump(detect_time_log, open(os.path.join(result_folder, detect_t_json_filename), 'w'))

        # write proposals
        region_json_filename = v + '_proposals.json' 
        json.dump(regions, open(os.path.join(result_folder, region_json_filename), 'w'))

        # write recognition results into json file 
        detect_json_filename = v + '_detections.json'
        json.dump(blob, open(os.path.join(result_folder, detect_json_filename), 'w'))

