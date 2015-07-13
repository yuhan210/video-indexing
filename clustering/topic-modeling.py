'''
Hierarchical topic modeling
'''
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import ward, dendrogram

import matplotlib
import matplotlib.pyplot as plt
try:
    plt.style.use('ggplot')
except:
    pass
import os
import lda
import sys
import json
import numpy as np



def load_video_summary(summary_folder, video_name):

    file_pref = os.path.join(summary_folder, video_name)

    # load caption
    with open(file_pref + '_5_caption.json') as json_file:
        caption_data = json.load(json_file)

    # sort caption results based on frame number
    caption_data = sorted(caption_data['imgblobs'], key=lambda x:int(x['img_path'].split('/')[-1].split('.')[0]))

    # load recognition
    with open(file_pref + '_recog.json') as json_file:
        recog_data = json.load(json_file)

    recog_data = sorted(recog_data['imgblobs'], key=lambda x: int(x['img_path'].split('/')[-1].split('.')[0]))

    # load blurinfo
    with open(file_pref + '_blur.json') as json_file:
        blur_data = json.load(json_file)

    blur_data = sorted(blur_data['img_blobs'], key=lambda x:int(x['img_name'].split('.')[0]))

    return caption_data, recog_data, blur_data



# input: a list of doc
# output: doc-term matrix
def getDocTermMatrix(corpus):

    vc = CountVectorizer(stop_words='english')
    dtm = vc.fit_transform(corpus).toarray()

    #print vc.vocabulary_ {"word":index}
    vocab =  vc.get_feature_names()
    
    return vocab, dtm
    #print dtm.shape
    #print vc.fit_transform(corpus)



def getTdidfMatrix(corpus):

    tv = TfidfVectorizer(stop_words='english') 
    tfidfm = tv.fit_transform(corpus)

    vocab = tv.get_feature_names()

    return vocab, tfidfm 


def plotTopicWordHist(topic_word):
    (n_topics, n_vocab) = topic_word.shape
    
    f, ax = plt.subplots(n_topics,1)
    for i in xrange(n_topics):
        ax[i].stem(topic_word[i,:]) 
        ax[i].set_ylim(0, 0.5)
        ax[i].set_title('topic {}'.format(i))

    ax[n_topics-1].set_xlabel('Word Prob')
    plt.tight_layout()
    plt.show()

def plotDocTopic(doc_topic):
    ## TODO: not done yet..
    (n_doc, n_topic) = doc_topic.shape
    
    f, ax = plt.subplots() 


# ward clustering algorithm
def hierarchicalClustering():
    print 'hi'   
 

def kmeans(tfidfm, n_clusters = 4):
    
    km = KMneas(n_clusters = n_clusters) 
    km.fit(tfidfm)    
    clusters = km.labels_
    
    print clusters.tolist()

def LDA(dtm, vocab, n_topics = 4):
       
 
    model = lda.LDA(n_topics=n_topics, n_iter=500, random_state=1)
    model.fit(dtm)
    
    topic_word = model.topic_word_
       
 
    # get the top 3 words for each topic (by probability)
    n = 10
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
        print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))

    plotTopicWordHist(topic_word)
    doc_topic = model.doc_topic_
    # N * n_topics histogram
    #print doc_topic
    #print("type(doc_topic): {}".format(type(doc_topic)))
    #print("shape: {}".format(doc_topic.shape))
    topic_most_pr = []
    for n in range(doc_topic.shape[0]):
        topic_most_pr += [doc_topic[n].argmax()]
        #print("doc: {} topic: {}".format(n, topic_most_pr))

    plt.figure()
    plt.plot(xrange(doc_topic.shape[0]), topic_most_pr, '.')
    plt.ylim([-1, n_topics])
    plt.show()


if __name__ == "__main__":

    if (len(sys.argv)) != 2:
        print 'Usage:', sys.argv[0], 'video_name'
        exit(-1)

    video_name = sys.argv[1]
    summary_folder = "/home/t-yuche/frame-analysis/video-summary"
     
    caption, recog, blud = load_video_summary(summary_folder, video_name)
    corpus = []
    for img_caption in caption:
        ## TODO: wieghted caption (merge 5 captions)
        doc = ' '.join(img_caption['candidate']['text'])
        #for cap in img_caption['candidate']['text']:
        corpus += [doc] 

    vocab, dtm = getDocTermMatrix(corpus)
    LDA(dtm, vocab) 
