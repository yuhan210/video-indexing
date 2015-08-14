import threading
import Queue
import pickle
import math
import time
import os
import sys
import cv2
CLUSTER_PATH = 'D:\panorama\clusterLib'
sys.path.append(CLUSTER_PATH)
from cluster import *
import numpy as np


#VIDEO_FOLDER = "/home/t-yuche/deep-video/data/videos"
VIDEO_FOLDER = "D:\\panorama\\videos"

event_queue = []

class RankedVideoPlayer(threading.Thread):

    def __init__(self, threadID, ranked_videos, stop_event, query_str):
        threading.Thread.__init__(self)
        self.threadID = threadID
        # (video_name, best_node, score)
        self.ranked_videos = ranked_videos
        self.stop_event = stop_event
        self.query_str = query_str

    def stop(self, caps):
        
        for cap in caps:
            cap.release()

        cv2.destroyAllWindows()
        return

    def init_videos(self, video_names, ranked_videos):
        caps = []
        video_chunks = []
        for idx, video in enumerate(video_names):

            if video.find('.mp4') < 0:
                video += '.mp4'

            video_path = os.path.join(VIDEO_FOLDER, video)           
            cap = cv2.VideoCapture(video_path)
            caps.append(cap)

            video_chunks += [(max(0, min(ranked_videos[idx][1]['n_idx'])), min(int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)), max(ranked_videos[idx][1]['n_idx'])))]

        return caps, video_chunks

    def run(self):
        COLUMN_NUM = 5

        video_names = [x[0] for x in self.ranked_videos]
        caps, video_chunks = self.init_videos(video_names, self.ranked_videos) 
   

        #check if videos are opened - should always work
        for idx, cap in enumerate(caps):
            if (not cap.isOpened()):
                print video_names[idx], 'not opened'
                cv2.destroyAllWindows() 
                return
            else:
                cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, video_chunks[idx][0])

        #stagger counter
        hashed_value = int(abs(hash(self.query_str)))
        stagger_counter = []
        for idx, dummy in enumerate(caps):
            
            if idx == 0:
                delayed_counter = hashed_value % 10
            else:
                delayed_counter = (hashed_value / math.pow(10, idx)) % 10

            stagger_counter += [delayed_counter * 15]

        #play all videos
        frame_counters = [x[0] for x in video_chunks]
        
        while(not self.stop_event.is_set()):
                    
            n_playing_video = 0
            frames = []
            finish_count = 0
            for idx, cap in enumerate(caps):
                video_name = video_names[idx] 
                
                if frame_counters[idx] == video_chunks[idx][1]: # need to replay                    
                    cv2.destroyWindow(video_name)
                    finish_count += 1
                    continue
                    
               
                ## creating stagger effect
                if stagger_counter[idx] > 0:
                    stagger_counter[idx] -= 1 
                else:
                 
                    ret, frame = cap.read()
                    frame = cv2.resize(frame, (350, 300))
                    frames += [frame]                 
                    frame_counters[idx] += 1
                    n_playing_video += 1

            # plot frames
            vis = np.zeros([1200, 1990, 3])
            for f_id, f in enumerate(frames):
                h, w = f.shape[:2]
                f = f/255.0
                # position
                col = f_id % 5
                row = f_id / 5                
                vis[30 + row * 400:30 + row * 400 + h ,col * 400 :col * 400 + w , :] = f            
            
            
            if n_playing_video > 0:
                # put text
                
                #cv2.PutText(vis, str(n_playing_video) + ' matched videos from x videos', (40, 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
                cv2.imshow('Panorama: Live Streaming - ' + self.query_str, vis)
                cv2.moveWindow('Panorama: Live Streaming - ' + self.query_str, 0, 0)
                c = cv2.waitKey(33/n_playing_video)

            if finish_count >= len(caps):
                cv2.destroyAllWindows()
                break
            
        self.stop(caps) 
 
    

def rank_video(matched_list, score_thresh = 0.1):
     
    ranked_list = sorted(matched_list, key = lambda x:x[2], reverse=True)
    ranked_list = filter(lambda x: x[2] > score_thresh, ranked_list) 

    return ranked_list



if __name__ == "__main__":

   
    old_index_file = 'index.pickle'
    index_file = './tmp/50_index_file.pickle'
    
    # load index from index.pickle file
    indexes = {}
    with open(index_file, 'rb') as pickle_fh:
       indexes = pickle.load(pickle_fh)
   
    old_indexes = {} 
    with open(old_index_file, 'rb') as pickle_fh:
       old_indexes = pickle.load(pickle_fh)

    indexes.update(old_indexes)
    video_names = indexes.keys()
    #print indexes
    #print video_names
    print len(video_names)

    thread_stop_event = None 
    while True:
        query_str = (raw_input('$ '))
        if query_str.find('quit') >= 0 or query_str.find('exit') >= 0:
            if thread_stop_event != None: 
                thread_stop_event.set()
                play_video_thread.join()
                play_video_thread = None
            break
    
        else:
            # match against all videos
            matched_list = []
            ranked_list = []
            matched_list = match_indexes(indexes, query_str)
            ranked_list = rank_video(matched_list)
            
            #for l in ranked_list:
            #    plt.plot(range(len(l[3])),l[3])
            #    plt.show()
                
            if len(ranked_list) == 0:
                cv2.destroyAllWindows()
                print '0 matched videos'
            #event_queue[0].put(ranked_list)
            else:
                if thread_stop_event != None: 
                    thread_stop_event.set()
                    play_video_thread.join()
                    play_video_thread = None
                thread_stop_event = threading.Event()
                
                play_video_thread = RankedVideoPlayer(2, ranked_list[:15], thread_stop_event, query_str) 
                play_video_thread.start()

