from PIL import Image
from utils import *
import os
import pickle

if __name__ == "__main__":

    VIDEO = '/mnt/video_list.txt'

    videos = open(VIDEO).read().split()
    obj_detect_processingtime = {}
    for vid, video_name in enumerate(videos):
        print video_name

        frame_names = os.listdir(os.path.join('/mnt/frames', video_name))
        # edge box profiling
        # get image resolution
        filepath = os.path.join('/mnt/frames', video_name, frame_names[0])
        im = Image.open(filepath)
        (w, h) = im.size # (width,height) tuple
        resol = str(w) + 'x' + str(h)
        frame_names = sorted(frame_names, key = lambda x: int(x.split('.')[0]))
        if w == 640 and h == 360:
            exec_time = []
            #print frame_names
            for frame_name in frame_names:
                fid = frame_name.split('.')[0]
                bbox_path = os.path.join('/mnt/tags/edgebox-all/', video_name, fid +'.bbx')
                with open(bbox_path, 'r') as f:
                    time_line = float(f.readline().strip().split(' ')[-1])
                    exec_time += [time_line]

            with open('./log/' + video_name + '.pickle', 'wb') as fh:
                pickle.dump(exec_time, fh)
        else:
            continue


