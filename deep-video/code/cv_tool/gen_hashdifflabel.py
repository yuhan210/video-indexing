from frame_compare import dhash
from PIL import Image
import cv2
import os
import sys
import json

if __name__ == "__main__":

    video_folder = sys.argv[1]
    for v in os.listdir(video_folder):
        print v

        blob = {}
        blob['video_name'] = v
        blob['img_blobs'] = []

        frame_folder = os.path.join(video_folder, v)
        for idx, f in enumerate(os.listdir(frame_folder)):
            frame_path = os.path.join(frame_folder, f)
            img = Image.open(frame_path)
            img_blob = {}
            img_blob['img_name'] = f
            img_blob['dhash'] = -1
            
            if idx > 0:

                dst = dhash(prev_frame, img)
                img_blob['dhash'] = dst
            
            blob['img_blobs'].append(img_blob)
            prev_frame = img

        json_filename = v + '_dhash.json'
        json.dump(blob, open(os.path.join('/home/t-yuche/deep-video/data/video_summary', json_filename), 'w'))
