import pickle
import tarfile
import itertools
import os
from utils import *

SERVER_WINDOW_SIZE = 30 * 5
for i in xrange(0, 50): 

    for query in open('../gen_rank_data/double_query_yx').readlines():
        query = query.strip()
        file_path = '../gen_rank_data/label_videos/' + query + '_' + str(i) + '.pickle'
        if not os.path.exists(file_path):
            continue
        vs = pickle.load(open('../gen_rank_data/label_videos/' + query + '_' + str(i) + '.pickle'))
        #pairs = list(itertools.combinations(vs,2))
        for v in vs:
        #for pair in pairs:
            video = v['video_name'] + '_' + str(v['start_fid']) + '_' + str(v['end_fid']) + '.mp4'
            #print video_a + '&video_name_b=' + video_b
            out_video_path = os.path.join('/mnt/video-segs', video)
            fps, w, h = get_video_fps(v['video_name'] + '.mp4')
            if os.path.exists(out_video_path):
                continue
            outstr = '~/ffmpeg/ffmpeg -framerate ' + str(fps) + ' -start_number ' + str(v['start_fid']) + ' -i /mnt/frames/' + v['video_name'] + '/%d.jpg -vframes ' + str(SERVER_WINDOW_SIZE) +' -s 560x420 -vcodec libx264 -pix_fmt yuv420p -n ' + out_video_path
            print outstr 
            

