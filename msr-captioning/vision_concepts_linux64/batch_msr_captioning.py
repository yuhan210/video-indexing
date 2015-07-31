import cv2; #opencv
import numpy as np; #numpy
import _init_paths; #sets up paths with caffe and test_utils
import cPickle
import demo_test_utils as tutils 


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
    prototxt_file = 'demoData/mil_finetune.prototxt.deploy';
    model_file = 'demoData/snapshot_iter_240000.caffemodel';
    vocab_file = 'demoData/vocab_train.pkl';
    model = loadModel(prototxt_file, model_file, vocab_file);
    imgList = ['demoData/COCO_train2014_000000581795.jpg',
               'demoData/COCO_train2014_000000348805.jpg',
               'demoData/COCO_train2014_000000465101.jpg',
               'demoData/COCO_train2014_000000581921.jpg'];
    for imName in imgList:
        print 'testing ',imName
        mil_prob, sc = testImg(imName, model)
        print 'top detections for this image'
        printWordsWithProb(mil_prob, model) #this will print top detections
        print 'start printing attribs for the MELM model'
        printAttribs(sc, imName, model) #this will output top 30 words for the MELM model to use;
										 #output of this print statement can be redirected to a file for MELM (called the "attributes" file)

if __name__ == '__main__':
    main();
