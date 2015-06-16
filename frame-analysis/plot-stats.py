import sys
import os
import json
import matplotlib.pyplot as plt

video_list = sys.argv[1]
video_summary_folder = sys.argv[2]

videos = [x.strip() for x in open(video_list).readlines()]

for v in videos:
    # Periscope-_upload_Periscope_video_to_YouTube_blur.json
    blur_json = os.path.join(video_summary_folder, v + '_blur.json')

    ## read blur data
    with open(blur_json) as blur_file:
        blur_data = json.load(blur_file)

    blur_data = sorted(blur_data['img_blobs'], key=lambda x:int(x['img_name'].split('.')[0]))
    
    ## read hash data
    hash_json = os.path.join(video_summary_folder, v + '_dhash.json')
    with open(hash_json) as hash_file:
        hash_data = json.load(hash_file)
    
    hash_data = sorted(hash_data['img_blobs'], key=lambda x:int(x['img_name'].split('.')[0]))
    
    ## read video frame size
    stats_json = os.path.join(video_summary_folder, v + '_encoding.json')
    with open(stats_json) as stats_file:
        stats_data = json.load(stats_file)

    frame_size = [x['pkt_size'] for x in stats_data['frames']]    
    hash_measure  = [x['dhash'] for x in hash_data] 
    measure = [x['blur_measure'] for x in blur_data]

    plt.subplot(3,1,1)    
    plt.plot(range(len(measure)), measure)
    plt.subplot(3,1,2)    
    plt.plot(range(len(hash_measure)), hash_measure)
    plt.subplot(3,1,3)    
    plt.plot(range(len(frame_size)), frame_size)
    plt.title(v)
    plt.show()
