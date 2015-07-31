import cv2 #opencv
import numpy as np #numpy
import _init_paths #sets up paths with caffe and test_utils
import cPickle
import demo_test_utils as tutils 
import sys

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
        

def main():

    if len(sys.argv) != 2:
        print 'Usage:', sys.argv[0], ' task_file'
        exit(-1)

    task_file = sys.argv[1]
    img_names = open(task_file, 'r').read().splitlines()
    top_k = 5

    prototxt_file = 'demoData/mil_finetune.prototxt.deploy';
    model_file = 'demoData/snapshot_iter_240000.caffemodel';
    vocab_file = 'demoData/vocab_train.pkl';
    model = loadModel(prototxt_file, model_file, vocab_file);

    for imName in img_names:
        print 'testing ',imName
        mil_prob, sc = testImg(imName, model)

	    srt_inds = np.argsort(sc)[::-1]; #sort in descending order
    	words = model['vocab']['words'];
    	print '%s: '%(imName)
    	for i in range(top_k):
	    	print '{:s} ({:.2f}), '.format(words[srt_inds[i]], sc[srt_inds[i]]);

