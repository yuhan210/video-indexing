'''
Functions doing indexing on the image summaries (caption, recognition)

'''
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import json
import os
import numpy as np

'''
Load caption/recognition/blur info from the summary 
folder given a list of videos
'''
def load_video_summary(summary_folder, video_name):
    
    file_pref = os.path.join(summary_folder, video_name)

    # load caption
    with open(file_pref + '_caption.json') as json_file:
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


'''
String matching query
'''
def build_strmatch(frame_list_file, summary_folder):

    videos = [x.strip() for x in open(frame_list_file).readlines()]
    for v in videos:
        captions, recogs, blurs = load_video_summary(summary_folder, v)    
    
'''
Bag-of-word Text search
'''
def build_tfidf(query_str, frame_list_file, summary_folder):
    
    videos = [x.strip() for x in open(frame_list_file).readlines()]
    corpus = ['' for x in xrange(len(videos))]

    # load summary for each video
    for idx, v in enumerate(videos):
        captions, recogs, blurs = load_video_summary(summary_folder, v)    
 
        v_words = []
        ## fill in all the captions into the corpus
        for img_cap in captions:
            for cap in img_cap['candidate']['text']:
                #corpus[idx] += [str(cap)]
                v_words.extend(str(cap).split(' '))
                 
        ## fill in all the recognition results into the corpus
        for img_pred in recogs:
            for pred in img_pred['pred']['text']:
                v_words += [str(pred)]

        ## make a list of word into a sentence
        corpus[idx] = ' '.join(v_words)
    
    corpus += [query_str]
    
    '''
    vc = CountVectorizer(stop_words='english')
    dtm = vc.fit_transform(corpus).toarray()
    print dtm
    print dtm.shape
    print dtm[0][703]
    print vc.fit_transform(corpus)
    '''

    tf = TfidfVectorizer()
    
    tfidf_matrix = tf.fit_transform(corpus) # tf * idf value for each doc
    #print 'cosine scores - ', cosine_similarity(tfidf_matrix[len(corpus)-1:len(corpus)], tfidf_matrix) 
    cos_scores = cosine_similarity(tfidf_matrix[len(corpus)-1:len(corpus)], tfidf_matrix[0:len(corpus)-1]) 
    video_name = videos[np.argmax(cos_scores)]
    score = cos_scores[0,np.argmax(cos_scores)]

    #feature_names = tf.get_feature_names()
    #idf = tf.idf_
    
    return video_name, score

