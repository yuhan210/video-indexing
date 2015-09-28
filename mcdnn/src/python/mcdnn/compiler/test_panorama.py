import inflection
import argparse
import sys
sys.path.append('/home/t-yuche/mcdnn/caffe/python')
import caffe
sys.path.append('/home/t-yuche/lib')
from utils import *
from wordnet import *
import ConfigParser
import pickle
import json
import os
import time
import numpy as np
import sys
rootdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(rootdir, "core"))
from example import face_input_prepare_n
globalconfig = ConfigParser.ConfigParser()
globalconfig.read(os.path.join(rootdir, "config", "global.conf"))
print(os.path.join(rootdir, "config", "global.conf"))
model_dir = globalconfig.get("mcdnn", "model_dir")

def tlc_caffe_indexes():


    # load tlc wnid to label
    tlc_wtl = {}
    with open('/home/t-yuche/mcdnn/data/labels/imagenet/2012/synset_words_tlc.txt') as f:
        for line in f.readlines():
            segs = line.strip().split()
            tlc_label = int(segs[0])
            wnid = segs[1]
            tlc_wtl[wnid] = tlc_label
 
    # load caffe wnid to label
    caffe_wtl = {}
    with open('/home/t-yuche/mcdnn/data/labels/imagenet/2012/train_caffe.txt') as f:
        for line in f.readlines():
            segs = line.strip().split()
            wnid = segs[0].split('/')[0]
            caffe_label = int(segs[1])

            if wnid not in caffe_wtl:
                caffe_wtl[wnid] = caffe_label

    # for each wnid
    caffe_to_tlc = {}
    tlc_to_caffe = {}
    for wnid in tlc_wtl.keys():

        # get caffe_label and tlc label
        tlc_label = tlc_wtl[wnid]
        caffe_label = caffe_wtl[wnid]
        tlc_to_caffe[tlc_label] = caffe_label
        caffe_to_tlc[caffe_label] = tlc_label

    return caffe_to_tlc, tlc_to_caffe

def batch_predict(configfile, targets):

    config = ConfigParser.ConfigParser()
    config.read(configfile)

    model_name = config.get("net", "net")
    model_def = os.path.join(model_dir, "%s.prototxt" % model_name)
    pretrained = os.path.join(model_dir, "%s.caffemodel" % model_name)
    mean = config.get("net", "mean")
    dim = int(config.get("net", "image_dim"))
    raw_scale = int(config.get("net", "raw_scale"))

    net = caffe.Classifier(model_def, pretrained, channel_swap=(2,1,0), mean=np.load(mean), raw_scale=raw_scale, image_dims=(dim, dim), batch=1)
    caffe.set_phase_test()
    caffe.set_mode_gpu()

    net2 = caffe.Net(config.get("train", "target_test"), config.get("train", "target_model"), 1)
    with open(config.get("train", "target_list"), "rb") as f:
        reverse_index = pickle.load(f)

    layer_name = config.get("train", "target_layer")
   
    count = 0
    exec_times = []
    prepare_times = []
    for cur_idx, image_path in enumerate(targets):
        im = caffe.io.load_image(image_path)
        tic = time.time() 
        prepared = face_input_prepare_n(net, [im], False)
        out = net.forward(end=layer_name, **{net.inputs[0]: prepared})
        out2 = net2.forward_all(**{net2.inputs[0]: out[layer_name]})[net2.outputs[0]]
        toc = time.time()
        exec_times += [(toc - tic)]
        i = out2[0].argmax()
        if i == len(reverse_index):
            print -1
        else:
            print reverse_index[i]

    print 'Execution time(ms) mean:', np.mean(exec_times),'std:', np.std(exec_times, ddof = 1)

def getwnid(caffe_label):

    with open('/home/t-yuche/mcdnn/data/labels/imagenet/2012/train_caffe.txt') as fh:
        for l in fh:
            segs = l.strip().split()
            wnid = segs[0].split('/')[0]
            if caffe_label == int(segs[1]):
                return wnid 

def wnid_to_word():
    
    d = {}
    with open('/home/t-yuche/mcdnn/data/labels/imagenet/2012/synset_words_caffe.txt') as fh:
        for l in fh:
            wnid = l.strip()[:9]
            s = l.strip()[10:]
            d[wnid] = s

    return d


def is_label_correct(label, label_to_trace, target_list, all_gt_words):
 
    # others:
    if label == -1:
        for target_label in target_list:
            trace = label_to_trace[target_label]
            for gt_word in all_gt_words:
                for pp in trace:
                    for p in pp: 
                        if p.split(',')[0] == gt_word:
                            return False
        return True

    else:
        trace = label_to_trace[label]
        for gt_word in all_gt_words:
            for pp in trace:
                for p in pp:
                    if p.split(',')[0] == gt_word:
                        return True
        return False

def is_match_splabel(labels, label_to_trace, target_list, gt):

    # all matched gt words
    all_gt_words = []
    for ll in gt:
        for l in ll:
            if not l.find('none') >= 0:
                gt_word = inflection.singularize(l.split('-')[-1])
                all_gt_words += [gt_word]

    
    for label in labels:
        is_correct = is_label_correct(label, label_to_trace, target_list, all_gt_words)
        if is_correct:
            return True

    return False
         

def is_match_label(pred, gt):

    # all matched gt words
    all_gt_words = []
    for ll in gt:
        for l in ll:
            if not l.find('none') >= 0:
                gt_word = inflection.singularize(l.split('-')[-1])
                all_gt_words += [gt_word]

    #for pl in pred:
    for gt_word in all_gt_words:
        for p_trace in pred:
            for pp in p_trace:
                for p in pp:
                    if p.split(',')[0] == gt_word:
                        return True
 
    return False

def load_sp_model(configfile):
    # load model
    config = ConfigParser.ConfigParser()
    config.read(configfile)

    model_name = config.get("net", "net")
    model_def = os.path.join(model_dir, "%s.prototxt" % model_name)
    pretrained = os.path.join(model_dir, "%s.caffemodel" % model_name)
    mean = config.get("net", "mean")
    dim = int(config.get("net", "image_dim"))
    raw_scale = int(config.get("net", "raw_scale"))
    layer_name = config.get("train", "target_layer")

    net = caffe.Classifier(model_def, pretrained, channel_swap=(2,1,0), mean=np.load(mean), raw_scale=raw_scale, image_dims=(dim, dim), batch=1)
    caffe.set_phase_test()
    caffe.set_mode_gpu()

    net2 = caffe.Net(config.get("train", "target_test"), config.get("train", "target_model"), 1)

    return net, net2 

def batch_predict_accuracy(configfile, net, net2, video_name, _vgg_data, turker_labels):

    config = ConfigParser.ConfigParser()
    config.read(configfile)

    layer_name = config.get("train", "target_layer")
    with open(config.get("train", "target_list"), "rb") as f:
        reverse_index = pickle.load(f)

    wnidtotrace = wnid_traces_dict()
    wptownid = word_prefix_to_wnid_dict()
    wnidtolabel = wnid_to_label_dict()
    # get the ancestors the word
    label_to_trace = {}
    for target_label in reverse_index:
        caffe_wnid = getwnid(target_label)
        trace = wnidtotrace[caffe_wnid]
        label_to_trace[target_label] = trace

    ## Finish loading necessary models ##
    print label_to_trace    
    top_5_vgg_count = 0
    top_1_vgg_count = 0
    top_1_sp_count = 0
    exec_time = []
    for keyframe_name in turker_labels:
        keyframe_path = os.path.join('/mnt/frames/', video_name, keyframe_name)
        
        # get accuracy for vgg
        vgg_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == int(keyframe_name.split('.')[0]), _vgg_data)
        vgg_words = vgg_data[0]['pred']['text']
   
        # get vgg_word trace
        top_1_vgg_labels = []
        top_5_vgg_labels = []
        for k, w in enumerate(vgg_words):
            wnid = wptownid[w]

            label = wnidtolabel[wnid]
            if not (label in reverse_index):
                label = -1
             
            if k == 0:
                top_1_vgg_labels += [label]
            
            top_5_vgg_labels += [label]

        #print vgg_words
        #print turker_labels[keyframe_name]
        #print top_1_vgg_labels
        #print top_5_vgg_labels
        vgg_top1_is_match = is_match_splabel(top_1_vgg_labels, label_to_trace, reverse_index, turker_labels[keyframe_name])

        vgg_top5_is_match = is_match_splabel(top_5_vgg_labels, label_to_trace, reverse_index, turker_labels[keyframe_name])

        if vgg_top1_is_match:
            top_1_vgg_count += 1

        if vgg_top5_is_match:
            top_5_vgg_count += 1

        #print turker_labels[keyframe_name]
        #print 'top_5_vgg:', top_5_vgg_trace
        #print 'top_1_vgg:', top_1_vgg_trace
        
        # predict using specialized
        im = caffe.io.load_image(keyframe_path)
        layer_name = config.get("train", "target_layer")
        tic = time.time()
        prepared = face_input_prepare_n(net, [im], False)
        out = net.forward(end=layer_name, **{net.inputs[0]: prepared})
        out2 = net2.forward_all(**{net2.inputs[0]: out[layer_name]})[net2.outputs[0]]
        toc = time.time()
        exec_time += [toc - tic]
        i = out2[0].argmax()

        sp_labels = []
        if i == len(reverse_index):
            sp_labels += [-1]
        else:
            sp_labels += [reverse_index[i]]

        #print sp_labels
        sp_top1_is_match = is_match_splabel(sp_labels, label_to_trace, reverse_index, turker_labels[keyframe_name]) 

        print vgg_top1_is_match, vgg_top5_is_match, sp_top1_is_match        
        if sp_top1_is_match:
            top_1_sp_count += 1
        #print vgg_top1_is_match, vgg_top5_is_match, sp_top1_is_match
    print 'Top 5 vgg:', top_5_vgg_count/(len(turker_labels) * 1.0)         
    print 'Top 1 vgg:', top_1_vgg_count/(len(turker_labels) * 1.0)         
    print 'Top 1 sp:', top_1_sp_count/(len(turker_labels) * 1.0)
    print 'SP exec time (ms):', np.mean(exec_time) * 1000, '+-', np.std(exec_time, ddof = 1)  * 1000
    return top_5_vgg_count, top_1_vgg_count, top_1_sp_count, len(turker_labels)     
    
if __name__ == "__main__":

    configfile = sys.argv[1]
    video_list = './test_video.txt'
    net, net2 = load_sp_model(configfile)
    
    t5_vgg_c = 0
    t1_vgg_c = 0
    t1_sp_c = 0
    total_c = 0
    with open(video_list) as f:
        for l in f:
            video_name = l.strip()
            if not os.path.exists(os.path.join('/mnt/frames', video_name)) or not os.path.exists(os.path.join('/mnt/tags/vgg-classify-all', video_name + '_recog.json')) or not os.path.exists(os.path.join('/mnt/turker-labels', video_name)):
                continue

            outfile = './alexnet_log/' + video_name + '.log'
            print video_name
            ''' 
            if os.path.exists(outfile):
                
                fh = open(outfile)
                segs = fh.read().split()
                top_5_vgg_count = int(segs[0])
                top_1_vgg_count = int(segs[1])
                top_1_sp_count = int(segs[2])
                total = int(segs[3])
                t5_vgg_c += top_5_vgg_count 
                t1_vgg_c += top_1_vgg_count
                t1_sp_c += top_1_sp_count
                total_c += total
                
                print t5_vgg_c, t1_vgg_c, t1_sp_c, total_c
                fh.close()
                continue
            '''
            fh = open(outfile, 'w')
            frames = [os.path.join('/mnt/frames', video_name, x) for x in sorted(os.listdir(os.path.join('/mnt/frames', video_name)), key = lambda x: int(x.split('.')[0]))]
   
            vgg_data = load_video_recog('/mnt/tags/vgg-classify-all', video_name)
            turker_ds = load_video_turker('/mnt/turker-labels', video_name) 
            top_5_vgg_count, top_1_vgg_count, top_1_sp_count, total = batch_predict_accuracy(configfile, net, net2, video_name, vgg_data, turker_ds) 
            print top_5_vgg_count, top_1_vgg_count, top_1_sp_count, total
            fh.write(str(top_5_vgg_count) + ' ' + str(top_1_vgg_count) + ' ' + str(top_1_sp_count) + ' ' + str(total) )

            t5_vgg_c += top_5_vgg_count 
            t1_vgg_c += top_1_vgg_count
            t1_sp_c += top_1_sp_count
            total_c += total
            print t5_vgg_c, t1_vgg_c, t1_sp_c, total_c
            fh.close()

    print t5_vgg_c, t1_vgg_c, t1_sp_c, total_c
    print 'VGG top-5: {:.4f} ({:d}/{:d})'.format(t5_vgg_c/(total_c * 1.0), t5_vgg_c, total_c)
    print 'VGG top-1: {:.4f} ({:d}/{:d})'.format(t1_vgg_c/(total_c * 1.0), t1_vgg_c, total_c)
    print 'SP top-1: {:.4f} ({:d}/{:d})'.format(t1_sp_c/(total_c * 1.0), t1_sp_c, total_c)
