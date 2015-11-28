import os
import time
import pickle
import json
import math
import datetime
import logging
import flask
import operator
import inflection
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np

VIDEOS = open('/mnt/video_list.txt').read().split()

# Obtain the flask app object
app = flask.Flask(__name__)

#http://localhost:5000/search?query=test
@app.route('/search', methods=['GET'])
def search():

    query_str = flask.request.args.get('query','')
    logging.info('Query: %s', query_str)
 
    opt_text_vis_ranks = app.opt_search.search_text_vis(0, query_str)[:20]
    opt_text_ranks = app.opt_search.search_text(0, query_str)[:20]
    meta_ranks = app.meta_search.search_metadata(query_str)
   
 
    return query_str

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

    def __init__(self, ALL_TEXT_FOLDER = '/home/t-yuche/ranking/gen_rank_data/index'):

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
             
        for stream_name in VIDEOS:
            stream_txt = cur_idx[stream_name]['text'] 
            stream_vis = cur_idx[stream_name]['vis']
            text_score = getCosSimilarty(stream_txt, query_dict)    
            vis_score = visual_match(query_str, stream_vis)
            scores[stream_name] = 0.5 * text_score + 0.5 * vis_score

        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        print sorted_scores[:20]
        return sorted_scores
    
    def search_text(self, rand_idx, query_str):

        cur_idx = self.indexes[rand_idx] 
        query_segs = query_str.split()
        query_dict = {}
        scores = {}
        for word in query_segs:
            query_dict[word] = 1/float(len(query_segs))
             
        for stream_name in VIDEOS:
            stream_txt = cur_idx[stream_name]['text'] 
            stream_vis = cur_idx[stream_name]['vis']
            text_score = getCosSimilarty(stream_txt, query_dict)    
            scores[stream_name] = text_score 

        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

        return sorted_scores


#TODO
#class SpecializeSearch():

class SubsampleSearch():

    def __init__(self, ):

    def search_text_vis(self, rand_idx, query_str):


    def search_text(self, rand_idx, query_str):


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
            if value >= edges[i] and value < edges[i + 1]: 
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
            return self.get_score(value, dislikes)
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
        metadata_features['rating'] = get_type_score(stream_info['rating'], 'rating')
        
        #
        words = stream_info['description'].split()
        for w in words:
            w = inflection.singularize(w.lower())
            for q_w in query_words:
                if w.find(q_w) >= 0:
                    metadata_features['description'] = 1
                    break
        #
        metadata_features['viewcount'] = get_type_score(stream_info['viewcount'], 'viewcount')
        #
        metadata_features['dislikes'] = get_type_score(stream_info['dislikes'], 'dislikes')
        #
        metadata_features['likes'] = get_type_score(stream_info['likes'], 'likes')

        #
        for w in stream_info['keywords']:
            w = inflection.singularize(w.lower())
            for q_w in query_words:
                if w.find(q_w) >= 0:
                    metadata_features['keywords'] = 1

        
        # compute score
        score = 0
        if metadata_features['title'] or metadata_features['description'] or metadata_features['keywords']: 
            for key in metadata_features:
                score += metadata_features[key] 

        return score
             
def start_tornado(app, port=5000):

    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()



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
    #app.opt_search = OptimalSearch()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    start_from_terminal(app)
