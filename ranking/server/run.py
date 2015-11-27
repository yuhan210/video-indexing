import os
import time
import pickle
import math
import datetime
import logging
import flask
import operator
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
import cStringIO as StringIO
import urllib

INDEX_FOLDER = './'

# Obtain the flask app object
app = flask.Flask(__name__)

#http://localhost:5000/search?query=test
@app.route('/search', methods=['GET'])
def search():
    query_str = flask.request.args.get('query','')
    logging.info('Query: %s', query_str) 
    opt_ranks = app.opt_search.search(0, query_str)[:20]
    
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

    def __init__(self):

        self.indexes = []
        for i in xrange(1):
            with open(os.path.join(INDEX_FOLDER, 'all_vis_'+ str(0) + '.pickle')) as fh:
                self.indexes += [pickle.load(fh)]  

    def search(self, rand_idx, query_str):
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
            vis_score = visual_match(query_str, stream_vis)
            scores[stream_name] = 0.5 * text_score + 0.5 * vis_score

        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        print sorted_scores[:20]
        return sorted_scores

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
    app.opt_search = OptimalSearch()
    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    start_from_terminal(app)
