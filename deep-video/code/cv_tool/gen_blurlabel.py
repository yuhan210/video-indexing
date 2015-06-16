from blur_detect import getSobel 
import cv2
import os
import sys
import json
import time

if __name__ == "__main__":

    video_folder = sys.argv[1]
    for v in os.listdir(video_folder):
        print v
        blob = {}
        blob['video_name'] = v
        blob['img_blobs'] = []
        frame_folder = os.path.join(video_folder, v)
        for f in os.listdir(frame_folder):
            img_blob = {}
            img_blob['img_name'] = f

            frame_path = os.path.join(frame_folder, f)
            img = cv2.imread(frame_path)
            # 23ms
            tic = time.time()
            dst = getSobel(img)
            toc = time.time()            

            img_blob['sobel_time'] = (toc-tic)
            img_blob['blur_measure'] = dst.mean()
            blob['img_blobs'].append(img_blob)

        json_filename = v + '_blur.json'
        json.dump(blob, open(os.path.join('/home/t-yuche/frame-analysis/blur-measure/', json_filename), 'w'))
