from utils import *
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
    UNIMPORTANTWORD_THRESH = 0.5
    FIG_OUTPUT_FOLDER = '/home/t-yuche/scheduling/skew/anec-figs/word_thresh_' + str(UNIMPORTANTWORD_THRESH)

       
    videos = open(VIDEO_LIST).read().split()

    for vid, video_name in enumerate(videos):
        print video_name
        n_frames = get_video_frame_num(video_name)
        
        output_path = os.path.join(FIG_OUTPUT_FOLDER, video_name + '.pdf') 
        scatter_data = [] 
        with open(os.path.join(OPTIMAL_INPUT_FOLDER, video_name + '_' + str(UNIMPORTANTWORD_THRESH) + '.pickle')) as fh:
            optimal_data = pickle.load(fh)
      
        start_fid = 0
        all_words = [] 
        while True:
            
            if start_fid > n_frames - 1:
                break
        
            end_fid = min(start_fid + SERVER_STORAGE_FRAMES, n_frames) 
            key = str(start_fid) + '-' + str(end_fid)
            optimal_tf = optimal_data[key]   

            #word_appear[end_fid] = [optimal_tf.keys()]
            for w in optimal_tf.keys():
                if w not in all_words:
                    all_words += [w]


            for w in optimal_tf.keys():
                widx = all_words.index(w)
                scatter_data += [(start_fid, widx)] 

            start_fid += SLIDE_SIZE_FRAMES

        plt.figure()    
        plt.scatter([x[0] for x in scatter_data], [x[1] for x in scatter_data])
        plt.xlabel('Frame ID')
        plt.ylabel('Word ID')
        plt.savefig(output_path, bbox_inches = 'tight')
        #plt.show()
