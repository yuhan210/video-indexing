from utils import *
import numpy as np
import random

if __name__ == "__main__":

    N_SEED = 1000
    N_STREAMS = 1000000
    GREEDY_INPUT_FOLDER = '/home/t-yuche/admission-control/window-greedy-log-0.5'
    THRESH = 0.8

    videos = open('/mnt/video_list.txt').read().split()


    percentages = []
    stream_frame_num = {}
    greedy_log = {}
    for video_name in videos:

        frame_num = get_video_frame_num(video_name)
        stream_frame_num[video_name] = frame_num 
        
        # load greedy subsampled frames
        greedy_gt_path = os.path.join(GREEDY_INPUT_FOLDER, video_name +  '_' + str(THRESH) + '_gtframe.pickle')
        with open(greedy_gt_path) as gt_fh:
            selected_frame_obj = pickle.load(gt_fh)
            greedy_log[video_name] = selected_frame_obj['picked_f']

    while len(videos) < N_STREAMS:
        videos += videos

    for ite in xrange(N_SEED):
        # select n_streams out of videos
        selected_videos = random.sample(videos, N_STREAMS) 
        
        fid_dict = {}
        for video_name in videos: 
            # starting at random point
            start_fid = random.randint(0, stream_frame_num[video_name] - 1)
            fid_dict[video_name] = start_fid 

            
        active_stream = 0
        for video_name in videos:
            
            # get current fid
            cur_fid = fid_dict[video_name]
            '''
            if cur_fid >=  stream_frame_num[video_name]:
                break
            '''
            if cur_fid in greedy_log[video_name]:
                active_stream += 1
                
        percentages += [active_stream/(len(videos) * 1.0)]
        print 'iteration:', ite, np.mean(percentages)

    print np.mean(percentages) 
