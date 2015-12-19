from utils import *
import pickle
import matplotlib
import matplotlib.pyplot as plt
try:
    plt.style.use('ggplot')
except:
    pass

if __name__ == "__main__":

    OPTIMAL_INPUT_FOLDER = '/home/t-yuche/admission-control/eval/optimal-log/optimal-tf'
    VIDEO_LIST = '/mnt/video_list.txt'
    SERVER_STORAGE_FRAMES = 5 * 30 # 5 sec * 30 fps
    SLIDE_SIZE_FRAMES = 1 * 30 # 1 sec * 30 fps
    UNIMPORTANTWORD_THRESH = 0.2
    FIG_OUTPUT_FOLDER = '/home/t-yuche/scheduling/skew/anec-figs/word_thresh_' + str(UNIMPORTANTWORD_THRESH)

       
    videos = open(VIDEO_LIST).read().split()

    all_words = [] 
    for vid, video_name in enumerate(videos):
        print video_name
        n_frames = get_video_frame_num(video_name)
        
        output_path = os.path.join(FIG_OUTPUT_FOLDER, video_name + '.pdf') 
        with open(os.path.join(OPTIMAL_INPUT_FOLDER, video_name + '_' + str(UNIMPORTANTWORD_THRESH) + '.pickle')) as fh:
            optimal_data = pickle.load(fh)
      
        start_fid = 0
        while True:
            if start_fid > n_frames - 1:
                break
        
            end_fid = min(start_fid + SERVER_STORAGE_FRAMES, n_frames) 
            key = str(start_fid) + '-' + str(end_fid)
            optimal_tf = optimal_data[key]   

            for w in optimal_tf.keys():
                if w not in all_words:
                    all_words += [w]


            start_fid += SLIDE_SIZE_FRAMES
            print len(all_words)

    with open('dictionary.pickle', 'wb') as fh:
        pickle.dump(all_words,fh)
