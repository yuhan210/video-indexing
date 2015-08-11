import shutil
import sys
import os
TOOL_PATH = '/home/t-yuche/clustering/tools'
sys.path.append(TOOL_PATH)
from utils import *

if __name__ == "__main__":

    ALL_FRAME_FOLDER = '/mnt/frames'

    for video_name in os.listdir(ALL_FRAME_FOLDER):# for each video
        print video_name
        os.makedirs('/mnt/key-frames/' + video_name)
        keyframes = loadKeyFrames(video_name) 
                
        # copy keyframe from /mnt/frames to keyframe folder 
        for frame in keyframes:
            segs = frame.split('/')
            segs[2] = 'key-frames'
            dst_path = '/'.join(segs)
            shutil.copyfile(frame, dst_path)
