import cv2 #opencv
import numpy as np #numpy
import _init_paths #sets up paths with caffe and test_utils
import cPickle
import demo_test_utils as tutils 
import sys
import time
import json
TOOL_PATH = '/home/t-yuche/clustering/tools' 
sys.path.append(TOOL_PATH)
from utils import * 

def loadModel(prototxt_file, model_file, vocab_file):
    means = np.array([[[ 103.939, 116.779, 123.68]]]);
    base_image_size = 565;
    with open(vocab_file, 'rb') as f:
       vocab = cPickle.load(f)
    model = tutils.load_model(prototxt_file, model_file, base_image_size, means, vocab);
    return model

def testImg(imname, model):
    im = cv2.imread(imname);
    net = model['net'];
    base_image_size = model['base_image_size'];
    means = model['means'];
    sc, mil_prob = tutils.test_img(im, net, base_image_size, means);
    return np.squeeze(mil_prob), np.squeeze(sc); #remove singleton dimensions
	
def printAttribs(sc, imname, model, topK=30):	
	srt_inds = np.argsort(sc)[::-1]; #sort in descending order
	words = model['vocab']['words'];
	print '%s: '%(imname)
	for i in range(topK):
		print '{:s} ({:.2f}), '.format(words[srt_inds[i]], sc[srt_inds[i]]);
	print '\n'


def printWordsWithProb(mil_prob, model, removeFunctional = False):
    functional_words = [];
    if removeFunctional:
        functional_words = ['a', 'on', 'of', 'the', 'in', 'with', 'and', 'is', 'to', 'an', 'two', 'at', 'next', 'are'];
    vocab = model['vocab'];
    words = vocab['words'];
    for i in range(len(words)):
        if words[i] not in functional_words:
            print mil_prob[i], words[i];
        

def loadProcessedTags(video_name):
    MSR_CAPTION_FOLDER = '/mnt/tags/msr-caption-keyframe'
    msr_data = load_video_msr_caption(MSR_CAPTION_FOLDER, video_name)

    return msr_data 

def loadKeyFrames(video_name):
    KEYFRAME_FOLDER = '/home/t-yuche/gt-labeling/frame-subsample/keyframe-info'
    keyframe_file = os.path.join(KEYFRAME_FOLDER, video_name + '_uniform.json')
    
    with open(keyframe_file) as json_file:
        keyframes = json.load(json_file)

    return keyframes 

if __name__ == "__main__":

    FRAME_FOLDER = '/mnt/frames'
    MSR_FOLDER = '/mnt/tags/msr-caption-all'
    top_k = 60
    prototxt_file = 'demoData/mil_finetune.prototxt.deploy';
    model_file = 'demoData/snapshot_iter_240000.caffemodel';
    vocab_file = 'demoData/vocab_train.pkl';
    model = loadModel(prototxt_file, model_file, vocab_file);
    
    for video_name in os.listdir(FRAME_FOLDER):
        print video_name 
    
        msr_data = loadProcessedTags(video_name)
        processed_frames = [x['img_path'] for x in msr_data]
        
        outfile_path = os.path.join(MSR_FOLDER, video_name + '_msrcap.json')

        if os.path.exists(outfile_path):
            print 'skip', video_name
            continue
        
        frame_folder = os.path.join(FRAME_FOLDER, video_name)
        
        blob = {}
        blob['imgblobs'] = []
        blob['imgblobs'] = blob['imgblobs'] + msr_data
        for frame_name in os.listdir(frame_folder):
            imName = os.path.join(frame_folder, frame_name)
            
            if imName in processed_frames:       
                continue

            tic = time.time()
            mil_prob, sc = testImg(imName, model)
            toc = time.time()
            srt_inds = np.argsort(sc)[::-1]; #sort in descending order
            words = model['vocab']['words'];
        
            texts = []
            probs  = []  
            for i in range(top_k):
                texts += [words[srt_inds[i]]]
                probs += [float(sc[srt_inds[i]])]
        
            img_blob = {}
            img_blob['img_path'] = imName
            img_blob['caption_time'] = (toc-tic)
            img_blob['words'] = {'text': texts, 'prob':probs }
            blob['imgblobs'].append(img_blob)

        print 'writing predictions to %s...' % (outfile_path, )
        json.dump(blob, open(outfile_path, 'w'))


