import random
import pickle

videos = open('/mnt/video_list.txt').read().split()
start_fid = {}
for video_name in videos:
    start_fid[video_name] = random.randint(0, 5 * 60 * 30)

with open('sim_start_fids.pickle', 'wb') as fh:
    pickle.dump(start_fid, fh)
