import pickle
import itertools

for i in xrange(1, 6): 

    for query in open('../gen_rank_data/single_query').read().split():
        vs = pickle.load(open('../gen_rank_data/label_videos/' + query + '_' + str(i) + '.pickle'))
        pairs = list(itertools.combinations(vs,2))
        for pair in pairs:
            video_a =  pair[0]['video_name'] + '_' + str(pair[0]['start_fid']) + '_' + str(pair[0]['end_fid']) + '.mp4'
            video_b =  pair[1]['video_name'] + '_' + str(pair[1]['start_fid']) + '_' + str(pair[1]['end_fid']) + '.mp4'
            print video_a + '&video_name_b=' + video_b + '&query=' + query + '&rand_idx=' + str(i)

    '''        
    for query in open('./double_query_yx').readlines():
        query = query.strip()
        vs = pickle.load(open('./label_videos/' + query + '_' + str(i) + '.pickle'))
        pairs = list(itertools.combinations(vs,2))
        for pair in pairs:
            video_a =  pair[0]['video_name'] + '_' + str(pair[0]['start_fid']) + '_' + str(pair[0]['end_fid']) + '.mp4'
            video_b =  pair[1]['video_name'] + '_' + str(pair[1]['start_fid']) + '_' + str(pair[1]['end_fid']) + '.mp4'
            print video_a + '&video_name_b=' + video_b
    '''
