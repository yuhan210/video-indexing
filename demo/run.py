import threading
import os
import sys
import cv2

VIDEO_FOLDER = "/home/t-yuche/deep-video/data/videos"

def play_video(video_names):

    COLUMN_NUM = 5


    caps = []
    for idx, video_name in enumerate(video_names):
        video_path = os.path.join(VIDEO_FOLDER, video_name)
        caps.append(cv2.VideoCapture(video_path))
   
    
     
    #create windows
    for idx, video_name in enumerate(video_names):
        cv2.namedWindow(video_name)
        cv2.resizeWindow(video_name, 50, 60)
        col = idx % 5
        row = idx / 5 
        cv2.moveWindow(video_name, col * 400, row * 400)
         
    cv2.waitKey(-1)

    #open videos
    for idx, cap in enumerate(caps):
        if (not cap.isOpened()):
            return

    #play all videos
    while(True):
        for idx, cap in enumerate(caps):
            ret, frame = cap.read()
            
            if ret == True:
                frame = cv2.resize(frame, (350, 300))
                cv2.imshow(video_names[idx], frame)                
                cv2.resizeWindow(video_name, 50, 60)
                col = idx % 5
                row = idx / 5 
                cv2.moveWindow(video_name, col * 400, row * 400)
                
        c = cv2.waitKey(33)
        if (c == 27):
            break
   
    cv2.destroyAllWindows() 
    

class VideosPlayerThread(threading.Thread):

    def __init__(self, threadID, video_names, stop_event):
        threading.Thread.__init__(self)  
        self.threadID = threadID
        self.video_names = video_names
        self.stop_event = stop_event
                

    def run(self):
        COLUMN_NUM = 5

        caps = []
        video_lengths = []
        for idx, video_name in enumerate(self.video_names):
            video_path = os.path.join(VIDEO_FOLDER, video_name)

            cap = cv2.VideoCapture(video_path)
            caps.append(cap)

            video_length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            video_lengths.append(video_length)
   
        #create windows
        for idx, video_name in enumerate(self.video_names):
            cv2.namedWindow(video_name)
            cv2.resizeWindow(video_name, 50, 60)
            col = idx % 5
            row = idx / 5 
            cv2.moveWindow(video_name, col * 400, row * 400)
         
        cv2.waitKey(-1)

        #check if videos are opened - should always work
        for idx, cap in enumerate(caps):
            if (not cap.isOpened()):
                print self.video_names[idx], 'not opened'
                cv2.destroyAllWindows() 
                return

        #play all videos
        frame_counters = [0 for x in xrange(len(self.video_names))]
        while(not self.stop_event.is_set()):
            
            for idx, cap in enumerate(caps):
                video_name = self.video_names[idx] 
                
                if frame_counters[idx] == video_lengths[idx]: # it is the last frame
                    frame_counters[idx] = 0
                    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
               
                  
                ret, frame = cap.read()
                frame = cv2.resize(frame, (350, 300))
                cv2.imshow(video_name, frame)                
                cv2.resizeWindow(video_name, 50, 60)
                col = idx % 5
                row = idx / 5 
                cv2.moveWindow(video_name, col * 400, row * 400)
 
                frame_counters[idx] += 1

            c = cv2.waitKey(33)
   
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows() 
    


class VideoPlayerThread(threading.Thread):

    def __init__(self, threadID, video_name, video_path):
        threading.Thread.__init__(self)  
        self.threadID = threadID
        self.video_name = video_name
        self.video_path = video_path

   
    def run(self): 
        print "Starting playing" , self.video_name
        window_name = self.video_name
        print window_name
        #cv2.namedWindow(window_name)
        cap = cv2.VideoCapture(self.video_path)

        if (not cap.isOpened()):
            return
        while (True):
        
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.resize(frame, (350, 300))
                cv2.imshow(video_names[idx], frame)                
                col = self.threadID % 5
                row = self.threading / 3 
                cv2.moveWindow(video_name, col * 400, row * 400)
                
                key = cv2.waitKey(33)  
                if key == 27: # ESC
                    break
            

        cv2.destroyAllWindows()


'''
class VideoPlayerWindowThread(threading.Thread):

    def __init__(self, threadID, video_names):
       
    def run(self)
'''

if __name__ == "__main__":

    if (len(sys.argv) != 2):
        print 'Usage:', sys.argv[0], 'video_names'
        exit(-1)
    
    video_names = [line.strip() for line in open(sys.argv[1]).readlines()]
    #thread_lock = threading.Lock()

    thread_stop_event = threading.Event()
    thread = VideosPlayerThread(1, video_names, thread_stop_event)
    thread.start()
    
    while True:
        
        query_str = (raw_input('$ ')) 
        if query_str == 'quit' or query_str == 'exit':
            thread_stop_event.set()
            break
    '''
    threads = []
    
    for idx, video_name in enumerate(video_names):
        threads.append(VideoPlayerThread(idx, video_name, os.path.join(VIDEO_FOLDER, video_name)))
    
    print threads 
    for idx, video_name in enumerate(video_names):
        threads[idx].start()
    '''
