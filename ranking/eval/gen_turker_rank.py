import os
import csv
import itertools
import pickle

def load_turker_video_labels(file_name):
    csv_file = open(file_name)
    csv_reader = csv.DictReader(csv_file, delimiter="\t") 
   
    selection_info = []
    for row in csv_reader:
        video_name_a = row['Answer.video_name_a']
        video_name_b = row['Answer.video_name_b']
        query_str = row['Answer.query_str']
        selection = row['Answer.selected_value'] 
        if len(selection) == 0:
            continue

        key = video_name_a + '_' + video_name_b
        selection_info += [{'va': video_name_a, 'vb':video_name_b, 'selection': selection, 'query_str':query_str}]


    return selection_info

def count(selections):

    video_names = list(set([x['va'] for x in selections] + [x['vb'] for x in selections]))
    select_count = {}
    for x in video_names:
        select_count[x] = 0
    for video_name in video_names:
        for selection in selections:
            if selection['va'] == video_name and selection['selection'] == 'A':
                select_count[video_name] += 1
          
            if selection['vb'] == video_name and selection['selection'] == 'B':
                select_count[video_name] += 1
        #print video_name, select_count[video_name] 

    #print select_count

def get_rank_score(rank, selections):

    satisfy_count = 0
    violate_videos = []
    for selection in selections:
        if selection['va'] in rank and selection['vb'] in rank:
            idx_a = rank.index(selection['va'])
            idx_b = rank.index(selection['vb'])

            if selection['selection'] == 'A': 
                if idx_a < idx_b:
                    satisfy_count += 1
                else:
                    if selection['va'] not in violate_videos:
                        violate_videos += [selection['va']]
                    if selection['vb'] not in violate_videos:
                        violate_videos += [selection['vb']]
            else:
                if idx_a > idx_b:
                    satisfy_count += 1
                else:
                    if selection['va'] not in violate_videos:
                        violate_videos += [selection['va']]
                    if selection['vb'] not in violate_videos:
                        violate_videos += [selection['vb']]

    #print '-----', rank, satisfy_count
    return satisfy_count, violate_videos

'''
def get_rank_score(rank, selections):

    violate_count = len(selections)
    for selection in selections:
        if selection['va'] in rank and selection['vb'] in rank:
            idx_a = rank.index(selection['va'])
            idx_b = rank.index(selection['vb'])

            if selection['selection'] == 'A': 
                if idx_a > idx_b:
                    violate_count += 1
            else:
                if idx_a < idx_b:
                    violate_count += 1
    return violate_count
'''
def get_best_position(rank, video_name, selections):
    _rank = list(rank)
    max_score = 0
    max_position = -1
    for i in xrange(len(rank) + 1):
        tmp_rank = list(_rank)
        tmp_rank.insert(i, video_name)
        score, dummy = get_rank_score(tmp_rank, selections)
        
        if score > max_score:
            max_position = i
            max_score = score

    return max_score, max_position

def get_betterone(video_a, video_b, selections):
    for selection in selections:
        if (selection['va'] == video_a and selection['vb'] == video_b):
            if selection['selection'] == 'A':
                return 0
            else:
                return 1
        elif (selection['va'] == video_b and selection['vb'] == video_a):
            if selection['selection'] == 'A':
                return 1
            else:
                return 0

def rank(selections):

    video_names = list(set([x['va'] for x in selections] + [x['vb'] for x in selections]))
    rank = [video_names[0]] 

    while len(rank) < len(video_names):
        max_score = -1
        max_position = [-1, -1]
        for video_name in video_names: # pick the best video
            if video_name not in rank:  
                score, position = get_best_position(rank, video_name, selections) 
                #print video_name, position, score
                if score > max_score:
                    max_score = score
                    max_position = [video_name, position]
                elif score == max_score and position == max_position[1]:
                    
                    b_idx = get_betterone(max_position[0], video_name, selections)
                    if b_idx == 1:
                        max_score = score
                        max_position = [video_name, position]
 

        #print 'inserting', max_position[0], 'in position', max_position[1]
        rank.insert(max_position[1], max_position[0])

    return rank

def swapping(rank, selections):
    print 'SWAPPING'

    score, bad_vs = get_rank_score(rank, selections)
    indexes = [rank.index(x) for x in bad_vs]
    min_idx = min(indexes) - 1
    max_idx = max(indexes) + 1
    #print max_idx - min_idx + 1
     
    for p in itertools.permutations(range(min_idx, max_idx + 1)):
    
        tmp_rank = list(rank)
        for c, i in enumerate(xrange(min_idx, max_idx+1)):
            tmp_rank[i] = rank[p[c]]
        score, dummy = get_rank_score(tmp_rank, selections)
        if score == 45:
            return tmp_rank
            break
    

def bruteforce(selections):
    video_names = list(set([x['va'] for x in selections] + [x['vb'] for x in selections]))

    n_videos = len(video_names)
    for p in itertools.permutations(range(n_videos)):
        rank = [video_names[x] for x in p]  
        score, dummy = get_rank_score(rank, selections)
        if score == 45:
            break 
        print p, score

if __name__ == "__main__":

    LABEL_FOLDER = '/home/t-yuche/ranking/process/process-labels/labels'
    for f in os.listdir(LABEL_FOLDER):

        if f.find('single') >= 0:
            ssid = f.split('.')[0].split('_')[-1]
            selection_info = load_turker_video_labels(os.path.join(LABEL_FOLDER, f))
      
            queries = list(set([x['query_str'] for x in selection_info]))

            for query in queries:
                selections = filter(lambda x: x['query_str'] == query, selection_info)
                out_rank = rank(selections)
                print query, get_rank_score(out_rank, selections)

                outputpath = os.path.join('./opt-rank', query + '_' + ssid + '.pickle')
                with open(outputpath, 'wb')  as fh:
                    pickle.dump(out_rank, fh)
