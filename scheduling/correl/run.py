import random

if __name__ == "__main__":

    N_SEED = 1000000
    N_STREAMS = 1000000

    videos = open('/mnt/video_list.txt').read().split()

    while len(videos) < N_STREAMS:
        videos += videos

    for ite in xrange(N_SEED):
        
        # select n_streams out of videos
        selected_videos = random.sample(videos, N_STREAMS) 
        
        fid =     
