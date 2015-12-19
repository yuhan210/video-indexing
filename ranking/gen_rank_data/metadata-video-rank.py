import pickle
import json 
import os
import inflection
import operator

VIDEOS = open('/mnt/video_list.txt').read().split()

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
            score, view_count = self.compute_metadata_score(self.metadata[stream_name], query_str)
            scores[stream_name] = (score, view_count)


        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse =True)
        title_scores = list(set(map(lambda x: x[1][0], sorted_scores)))
        title_scores.sort(reverse = True)
    
        sorted_viewcount = []
        for title_score in title_scores:
            scs = filter(lambda x: x[1][0] == title_score, sorted_scores)    
            scs = sorted(scs, key=lambda x:x[1][1], reverse =True)
            sorted_viewcount += scs

        print sorted_scores
        return sorted_scores

    def get_score(self, value, edges):
        
        for i in xrange(len(edges) - 1):
            if value >= edges[i] and value <= edges[i + 1]: 
                return (i+1)/((len(edges) - 1) * 1.0)

    def get_type_score(self, value, value_type, TYPE = 0):

        if TYPE == 0:
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

        else:
            if value_type == 'viewcount': 
                return value
            elif value_type == 'rating': 
                return 0
            elif value_type == 'dislikes':
                return 0
            elif value_type == 'likes': 
                return 0
            

    def compute_metadata_score(self, stream_info, query_str): 
        TYPE = 1

        metadata_features = {'title': 0, 'rating': -1, 'description': 0, 'viewcount': -1, 'dislikes':-1, 'likes': -1,  'keywords': 0}
        query_words = query_str.split()

        # 
        words = stream_info['ftitle'].split('_')
        for q_w in query_words:
            for w in words:
                w = inflection.singularize(w.lower())
                if w.find(q_w) >= 0:
                    metadata_features['title'] += 1
                    break
        #
        metadata_features['rating'] = self.get_type_score(stream_info['rating'], 'rating', TYPE)
        
        #
        words = stream_info['description'].split()
        for q_w in query_words:
            for w in words:
                w = inflection.singularize(w.lower())
                if w.find(q_w) >= 0:
                    metadata_features['description'] += 1
                    break
        #
        metadata_features['viewcount'] = self.get_type_score(stream_info['viewcount'], 'viewcount', TYPE)
        #
        metadata_features['dislikes'] = self.get_type_score(stream_info['dislikes'], 'dislikes', TYPE)
        #
        metadata_features['likes'] = self.get_type_score(stream_info['likes'], 'likes', TYPE)

        #
        for q_w in query_words:        
            for w in stream_info['keywords']:
                w = inflection.singularize(w.lower())
                if w.find(q_w) >= 0:
                    metadata_features['keywords'] += 1
                    break
        
        # compute score
        score = 0
        if metadata_features['title'] or metadata_features['description'] or metadata_features['keywords']: 
            score = metadata_features['title'] + metadata_features['description'] + metadata_features['keywords']
            
            #for key in metadata_features.keys():
            #    score += metadata_features[key] 

        return score, metadata_features['viewcount']



if __name__ == "__main__":

    METADATA_KEY_RESULTS_FILE = './metadata_key_results.pickle'
    with open('./start_fid_set.pickle') as fh:
        start_fid_set = pickle.load(fh)

    #with open('./video_frame_num.pickle') as fh:
    #    video_framenum = pickle.load(fh)

    metadata_rank = {}
    ''' 
    if os.path.exists(METADATA_KEY_RESULTS_FILE):
        with open(METADATA_KEY_RESULTS_FILE) as fh:
            metadata_rank = pickle.load(fh)
    '''

     
    metadata_search = MetadataSearch('./video-info') 
    for query in open('./query').readlines():
        query = query.strip()
        metadata_rank[query] = metadata_search.search_metadata(query)


           
    with open(METADATA_KEY_RESULTS_FILE, 'wb') as fh: 
        pickle.dump(metadata_rank, fh) 
