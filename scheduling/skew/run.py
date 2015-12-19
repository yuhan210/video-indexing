from utils import *
import matplotlib
import matplotlib.pyplot as plt
try:
    plt.style.use('ggplot')
except:
    pass
font = {
        'size'   : 24,
    }
text = {'color': 'black'}
line = {'linewidth': 3}
matplotlib.rc('font', **font)
matplotlib.rc('text', **text)
matplotlib.rc('lines', **line)


if __name__ == "__main__":

    OPTIMAL_INPUT_FOLDER = '/home/t-yuche/admission-control/eval/optimal-log/optimal-tf'
    #OPTIMAL_INPUT_FOLDER = '/home/t-yuche/admission-control/eval/optimal-log/optimal-tf-less-stopword'
    VIDEO_LIST = '/mnt/video_list.txt'
    SERVER_STORAGE_FRAMES = 5 * 30 # 5 sec * 30 fps
    SLIDE_SIZE_FRAMES = 1 * 30 # 1 sec * 30 fps
    UNIMPORTANTWORD_THRESH = 0.5
    #FIG_OUTPUT_FOLDER = '/home/t-yuche/scheduling/skew/anec-figs/word_thresh_lessstopword_' + str(UNIMPORTANTWORD_THRESH)
    FIG_OUTPUT_FOLDER = '/home/t-yuche/scheduling/skew/anec-figs/word_thresh_' + str(UNIMPORTANTWORD_THRESH)

       
    videos = open(VIDEO_LIST).read().split()

    for vid, video_name in enumerate(videos):
        if not (video_name == 'brave_pet_cat_stands_up_to_mountain_lion__cute_cats_vs_mountain_lion_Vnmxg4h_Mio' or video_name == 'crazy_cliff_driving_crash__bmw_m3_drives_off_cliff_az_Vlgy5dbM' or video_name == 'turtle_epically_eating_tomato_GvH4odJa8co' or video_name == 'hd_polar_bear_on_thin_ice__natures_great_events_the_great_melt__bbc_one_Kv9v9ALV3yk' or video_name == '70000_subscriber_thank_you_2bg_real_life_basketball_ep_5__game_of_horse_Qft3fOZjBgU') :
            continue
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
                scatter_data += [(start_fid, widx + 1)] 

            start_fid += SLIDE_SIZE_FRAMES

        plt.figure()    
        plt.scatter([x[0]/33 for x in scatter_data], [x[1] for x in scatter_data], marker = 'x')
        plt.xlabel('Time (sec)')
        plt.ylabel('Word ID')
        plt.xlim([-1, n_frames/33 + 1])
        plt.ylim([0, max([x[1] for x in scatter_data]) + 1])
        plt.savefig(output_path, bbox_inches = 'tight')
        #plt.show()
