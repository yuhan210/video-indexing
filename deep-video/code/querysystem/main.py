from queryProcessing.build_index import build_strmatch, build_tfidf
from utils.video import play_video

import sys
import cv2

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print 'Usage', sys.argv[0], ' video_summary_folder frame_list'
        exit(-1)

    video_summary_folder = sys.argv[1]
    frame_list_file = sys.argv[2]

    #build_strmatch(frame_list_file, video_summary_folder)
    #train_tfidf_matrix = build_tfidf(frame_list_file, video_summary_folder)

    while True:
        try:
            query_str = (raw_input('$ '))

        except ValueError:
            print('Invalid Qeury')
            continue

        if query_str == 'quit' or query_str == 'exit':
            break
        else:
            video_name, score = build_tfidf(query_str, frame_list_file, video_summary_folder)
            if score == 0.0:
                print 'Not found'
            else:
                print video_name, score
                play_video('/home/t-yuche/deep-video/data/videos/' + video_name + '.mp4', 5)
            #process_query()
    
