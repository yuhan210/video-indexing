import os
import time
import pickle
import json
import math
import datetime
import logging
import flask
from flask.ext.cors import CORS
import operator
import inflection
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
from utils import *

VIDEOS = open('/mnt/video_list.txt').read().split()
TOPK = 10
SERVER_STORAGE_FRAMES = 5 * 30

# Obtain the flask app object
app = flask.Flask(__name__)
CORS(app)

#http://localhost:5000/search?query=test
@app.route('/search', methods=['GET'])
def search():

    query_str = flask.request.args.get('query','')
    rand_idx = int(flask.request.args.get('rand_idx','0'))
    logging.info('Query: %s, rand_idx: %d', query_str, rand_idx)
 
    opt_text_vis_ranks = app.opt_search.search_text_vis(rand_idx, query_str)[:TOPK]
    opt_text_ranks = app.opt_search.search_text(rand_idx, query_str)[:TOPK]

    print opt_text_vis_ranks

    meta_ranks = app.meta_search.search_metadata(query_str)[:TOPK]
    print meta_ranks  
    response_dict = compose_response(rand_idx, opt_text_vis_ranks, opt_text_ranks, meta_ranks) 

    return flask.jsonify(success = True, response = response_dict) 

def get_videoseg_name(video_name, fid):

    n_frames = get_video_frame_num(video_name)

    chunk = fid / SERVER_STORAGE_FRAMES
    start_fid = chunk * SERVER_STORAGE_FRAMES
    end_fid = start_fid + SERVER_STORAGE_FRAMES

    videoseg_name = video_name + '_' + str(start_fid) + '_' + str(end_fid)  + '.mp4'

    return videoseg_name, start_fid, end_fid



def compose_response(rand_idx, opt_text_vis_ranks, opt_text_ranks, meta_ranks):

    response_dict = {}

    
    response_dict['opt_text_vis'] = {}
    for idx, tup in enumerate(opt_text_vis_ranks):

        stream_name = tup[0]
        p = [pos for pos, char in enumerate(stream_name) if char == '_']
        video_name = stream_name[:p[-2]]
        start_frame_fid = int(stream_name[p[-2]+1:p[-1]])
        end_frame_fid = int(stream_name[p[-1]+1:-4])
        start_time = start_frame_fid/int(stream_rates[video_name])
        end_time =  end_frame_fid/int(stream_rates[video_name])
        
        response_dict['opt_text_vis'][idx] = {'video_name': video_name, 'start_time': start_time, 'end_time': end_time, 'score': tup[1]}

    response_dict['opt_text'] = {}
    for idx, tup in enumerate(opt_text_ranks):
        stream_name = tup[0]
        p = [pos for pos, char in enumerate(stream_name) if char == '_']
        video_name = stream_name[:p[-2]]
        start_frame_fid = int(stream_name[p[-2]+1:p[-1]])
        end_frame_fid = int(stream_name[p[-1]+1:-4])
        start_time = start_frame_fid/int(stream_rates[video_name])
        end_time =  end_frame_fid/int(stream_rates[video_name])

        response_dict['opt_text'][idx] = {'video_name': video_name, 'start_time': start_time, 'end_time': end_time, 'score': tup[1]}

    response_dict['metadata'] = {}
    for idx, tup in enumerate(meta_ranks):
        video_name = tup[0]
        start_frame_fid = start_fid_set[rand_idx][video_name] 
        stream_name, start_fid, end_fid = get_videoseg_name(video_name, start_frame_fid)
        start_time = start_fid/int(stream_rates[video_name])
        end_time =  end_fid/int(stream_rates[video_name])

        response_dict['metadata'][idx] = {'video_name': video_name, 'start_time': start_time, 'end_time': end_time, 'score': tup[1]}

    return response_dict


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)


def getCosSimilarty(a_dict, b_dict):
    #print a_dict, b_dict
    space = list(set(a_dict.keys()) | set(b_dict.keys()))
    # compute consine similarity (a dot b/ |a| * |b|)
    sumab = 0.0
    sumaa = 0.0
    sumbb = 0.0

    for dim in space:
        a = 0.0
        b = 0.0
        if dim in a_dict.keys():
            a = a_dict[dim]
        if dim in b_dict.keys():
            b = b_dict[dim]

        sumab += a * b
        sumaa += a ** 2
        sumbb += b ** 2       
    
    return sumab/(math.sqrt(sumaa * sumbb))


def visual_match(query_str, vis_feature):
    
    words = query_str.split()
    word = words[0]
    if word in vis_feature.keys(): 
        dwell_time = vis_feature[word]['dwell_time']/150.
        obj_num = vis_feature[word]['obj_num']
        ave_obj_size = vis_feature[word]['ave_obj_size']
        max_obj_size = vis_feature[word]['max_obj_size']
        moving_speed = vis_feature[word]['moving_speed'] 

    return dwell_time 

class OptimalSearch():  

    def __init__(self, ALL_TEXT_FOLDER = '/home/t-yuche/ranking/gen_rank_data/optimal-index'):

        self.indexes = []
        for i in xrange(1):
            with open(os.path.join(ALL_TEXT_FOLDER, 'all_vis_'+ str(0) + '.pickle')) as fh:
                self.indexes += [pickle.load(fh)]  

    def search_text_vis(self, rand_idx, query_str):

        cur_idx = self.indexes[rand_idx] 
        query_segs = query_str.split()
        query_dict = {}
        scores = {}
        for word in query_segs:
            query_dict[word] = 1/float(len(query_segs))
             
        for stream_name in cur_idx.keys():
            stream_txt = cur_idx[stream_name]['text'] 
            stream_vis = cur_idx[stream_name]['vis']
            start_fid = cur_idx[stream_name]['start_fid']
            end_fid = cur_idx[stream_name]['end_fid']
            video_name = cur_idx[stream_name]['video_name']
            text_score = getCosSimilarty(stream_txt, query_dict)    
            vis_score = visual_match(query_str, stream_vis)
            scores[stream_name] = 0.5 * text_score + 0.5 * vis_score

        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_scores
    
    def search_text(self, rand_idx, query_str):

        cur_idx = self.indexes[rand_idx] 
        query_segs = query_str.split()
        query_dict = {}
        scores = {}
        for word in query_segs:
            query_dict[word] = 1/float(len(query_segs))
             
        for stream_name in cur_idx.keys():
            stream_txt = cur_idx[stream_name]['text'] 
            stream_vis = cur_idx[stream_name]['vis']
            text_score = getCosSimilarty(stream_txt, query_dict)    
            scores[stream_name] = text_score 

        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

        return sorted_scores


#TODO
#class SpecializeSearch():
'''
class SubsampleSearch():

    def __init__(self, ):

    def search_text_vis(self, rand_idx, query_str):


    def search_text(self, rand_idx, query_str):
'''

class MetadataSearch():

    def __init__(self, METADATA_FOLDER = '/mnt/video-info'):

        self.metadata = {}
       
        for stream_name in VIDEOS:
            with open(os.path.join(METADATA_FOLDER, stream_name + '.json')) as fh:
                data = json.load(fh)
                self.metadata[stream_name] = data 


    def search_metadata(self, query_str):

        scores = {}
        for stream_name in VIDEOS:
            score = self.compute_metadata_score(self.metadata[stream_name], query_str)
            scores[stream_name] = score  

        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_scores

    def get_score(self, value, edges):
        
        for i in xrange(len(edges) - 1):
            if value >= edges[i] and value <= edges[i + 1]: 
                return (i+1)/((len(edges) - 1) * 1.0)

    def get_type_score(self, value, value_type):
        ratings = [0.0, 4.42857122421, 4.78807926178, 4.90198802948, 4.97389554977,5.0]
        likes = [0, 6, 91, 650, 4002, 740417]
        dislikes = [209830, 186, 30, 4, 0, 0]
        viewcounts = [0, 1200, 20949, 160330, 1194433, 591549856]
        
        if value_type == 'viewcount': 
            return self.get_score(value, viewcounts)
        elif value_type == 'rating': 
            return self.get_score(value, ratings)
        elif value_type == 'dislikes':
            for i in xrange(len(dislikes) - 1):
                if value <= dislikes[i] and value >= dislikes[i+1]:
                    return (i+1)/((len(dislikes) -1) * 1.0)
        elif value_type == 'likes': 
            return self.get_score(value, likes)


    def compute_metadata_score(self, stream_info, query_str): 

        metadata_features = {'title': 0, 'rating': -1, 'description': 0, 'viewcount': -1, 'dislikes':-1, 'likes': -1,  'keywords': 0}
        query_words = query_str.split()

        # 
        words = stream_info['ftitle'].split('_')
        for w in words:
            w = inflection.singularize(w)
            for q_w in query_words:
                if w.find(q_w) >= 0:
                    metadata_features['title'] = 1
                    break

        #
        metadata_features['rating'] = self.get_type_score(stream_info['rating'], 'rating')
        
        #
        words = stream_info['description'].split()
        for w in words:
            w = inflection.singularize(w.lower())
            for q_w in query_words:
                if w.find(q_w) >= 0:
                    metadata_features['description'] = 1
                    break
        #
        metadata_features['viewcount'] = self.get_type_score(stream_info['viewcount'], 'viewcount')
        #
        metadata_features['dislikes'] = self.get_type_score(stream_info['dislikes'], 'dislikes')
        #
        metadata_features['likes'] = self.get_type_score(stream_info['likes'], 'likes')

        #
        for w in stream_info['keywords']:
            w = inflection.singularize(w.lower())
            for q_w in query_words:
                if w.find(q_w) >= 0:
                    metadata_features['keywords'] = 1

        
        # compute score
        score = 0
        if metadata_features['title'] or metadata_features['description'] or metadata_features['keywords']: 
            for key in metadata_features.keys():
                score += metadata_features[key] 

        return score
             
def start_tornado(app, port=5000):

    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def get_stream_seg(stream_name):
    with open('/home/t-yuche/ranking/gen_rank_data/start_fid_set.pickle') as fh:
        start_fid_set = pickle.load(fh) 

def start_from_terminal(app):

    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)

    opts, args = parser.parse_args()

    # load index
    app.meta_search = MetadataSearch()
    app.opt_search = OptimalSearch()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


def init():
    
    global stream_rates
    global start_fid_set
    stream_rates = {}

    for video_name in VIDEOS:
        fps, dummy, dummy = get_video_fps(video_name)
        stream_rates[video_name] = fps

    with open('start_fid_set.pickle') as fh:
        start_fid_set = pickle.load(fh)


if __name__ == '__main__':
    init()
    logging.getLogger().setLevel(logging.INFO)
    start_from_terminal(app)
