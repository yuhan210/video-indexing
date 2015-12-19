from utils import *
import matplotlib
import matplotlib.pyplot as plt
import sys
import time
from vision import *
from multiprocessing import Process, Queue
try:

    plt.style.use('ggplot')
except:
    pass
font = {
        'size'   : 8,
    }
text = {'color': 'black'}
line = {'linewidth': 3}
matplotlib.rc('font', **font)
matplotlib.rc('text', **text)
matplotlib.rc('lines', **line)


MV_INPUT_FOLDER = '/home/t-yuche/admission-control/train/mv_log'
def phash(cur_img, prev_img, que):
    pilcur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
    pilprev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)
    pil_cur = Image.fromarray(pilcur_img)
    pil_prev = Image.fromarray(pilprev_img)
    a_hash = imagehash.phash(pil_cur)
    b_hash = imagehash.phash(pil_prev)

    que.put(a_hash - b_hash)


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
        #if not (video_name == 'brave_pet_cat_stands_up_to_mountain_lion__cute_cats_vs_mountain_lion_Vnmxg4h_Mio' or video_name == 'crazy_cliff_driving_crash__bmw_m3_drives_off_cliff_az_Vlgy5dbM' or video_name == 'turtle_epically_eating_tomato_GvH4odJa8co' or video_name == 'hd_polar_bear_on_thin_ice__natures_great_events_the_great_melt__bbc_one_Kv9v9ALV3yk' or video_name == '70000_subscriber_thank_you_2bg_real_life_basketball_ep_5__game_of_horse_Qft3fOZjBgU') :
        #    continue
        #if not (video_name == 'waving_bear_r9bZMZw7aVQ'):
        #    continue
        n_frames = get_video_frame_num(video_name)
        if not video_name == sys.argv[1]:
            continue
        #if n_frames >= 2500 or video_name == '120121_kyuhyun_singing_with_sungmin_accompanying_him_guitar__spZZL2FJiMk': 
        #    continue
        print video_name
        #output_path = os.path.join(FIG_OUTPUT_FOLDER, video_name + '.pdf') 
        scatter_data = [] 
        with open(os.path.join(OPTIMAL_INPUT_FOLDER, video_name + '_' + str(UNIMPORTANTWORD_THRESH) + '.pickle')) as fh:
            optimal_data = pickle.load(fh)
        # read ml trace
        mltraceoath = os.path.join('/home/t-yuche/admission-control/train/multi-layer/leaveoneout/pred_trace', video_name + '.pickle')   
        
        with open(mltraceoath) as fh:
            ml_data = pickle.load(fh)
        ml_picked_fid = ml_data['picked_f']
 
        GREEDYFOLDER = '/home/t-yuche/admission-control/greedy/window-greedy-log-0.5'
        #GREEDYFOLDER = '/home/t-yuche/admission-control/greedy/window-greedy-log-multi-0.5'
        greedypath = os.path.join(GREEDYFOLDER, video_name + '_' + str(0.8)  + '_gtframe.pickle')
        gt_data = pickle.load(open(greedypath))
        gt_picked_fid = gt_data['picked_f']
        total_frame_n = gt_data['total_frame']

 
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

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8,ax9), (ax10, ax11, ax12)) = plt.subplots(4,3)
        ax1.scatter([x[0] for x in scatter_data], [x[1] for x in scatter_data], marker = 'x')
        ax1.set_xlabel('Frame ID')
        ax1.set_ylabel('Word ID')
        ax1.set_xlim([-1, n_frames])
        ax1.set_ylim([0, max([x[1] for x in scatter_data]) + 1])
        #plt.savefig(output_path, bbox_inches = 'tight')
        for pf in gt_picked_fid: 
            ax1.plot((pf, pf), (0, max([x[1] for x in scatter_data])), 'k-' )
        for pf in ml_picked_fid:
            ax1.plot((pf, pf), (0, max([x[1] for x in scatter_data])), 'r-' )

        encdata = getMetadata(video_name) 
        cvdata = getCVInfoFromLog(video_name)
        mv_file = os.path.join(MV_INPUT_FOLDER, video_name + '.pickle')
        with open(mv_file) as fh:
            mv_features = pickle.load(fh)
        
        frame_diffs = []
        phashs = []
        sift_scores = []
        surf_scores = []
        hist_scores = []
        enc_sizes = []
        mv_sizes = []
        ave_mv_sizes = []
        max_mv_sizes = []
        sobels = []
        illus = []
        illu_changes = []
        print 'Start'
        prev_framename = '0.jpg'
        prev_img = cv2.imread(os.path.join('/mnt/frames', video_name, prev_framename))
        prev_img = cv2.resize(prev_img, (160, 120))
        for i in xrange(1, n_frames):
            tic = time.time() 
            framename = str(i) + '.jpg'   
            cur_img = cv2.imread(os.path.join('/mnt/frames', video_name, framename))     
            org_h, org_w = cur_img.shape[:2]
            ### 
            cur_img = cv2.resize(cur_img, (160, 120)) 
            h, w = cur_img.shape[:2]
            
            enc = encdata[framename]  
            cv = cvdata[framename]
            if enc['type'] == 'I':
               enc_sizes += [-1]
            else:
               enc_sizes += [enc['size']]

            mv_sizes += [mv_features[i][0]/ (org_w* org_h * 1.0)]
            ave_mv_sizes += [mv_features[i][1]]
            max_mv_sizes += [mv_features[i][2]]
            sobels += [cv['sobel'][0]]
            if len(illus) > 0:
                illu_changes += [cv['illu'][0] - illus[-1]]
            else:
                illu_changes += [-1]
            illus += [cv['illu'][0]]
            

            queue_fd = Queue() 
            queue_ph = Queue() 
            queue_ch = Queue() 
            queue_si = Queue() 
            queue_su = Queue()
 
            pfd = Process(target = getFrameDiffq, args = (prev_img, cur_img,  queue_fd))
            pph = Process(target = phash, args = (cur_img, prev_img, queue_ph))
            pch = Process(target = colorHistSimq, args = (prev_img, cur_img, queue_ch))
            psi = Process(target = getSIFTMatchingSimq, args = (prev_img, cur_img, queue_si))
            psu = Process(target = getSURFMatchingSimq, args = (prev_img, cur_img, queue_su))

            pph.start()
            pfd.start()
            psi.start()
            pch.start()
            psu.start()

            pfd.join()
            pph.join()
            pch.join()
            psi.join()
            psu.join()
            frame_diffs += [queue_fd.get()/(h * w * 1.0)]   
            phashs += [queue_ph.get()/64.]
            hist_scores += [queue_ch.get()] 
            sift_scores += [queue_si.get()]
            surf_scores += [queue_su.get()]

            if i in gt_picked_fid:
                prev_framename = framename
                prev_img = cur_img.copy()
            toc = time.time()   
            print i, (toc-tic) * 1000

        ax2.scatter([x for x in range(1, n_frames)], [x for x in frame_diffs], marker = 'x')
        ax2.set_xlabel('Frame ID')
        ax2.set_ylabel('framediff')
        ax2.set_xlim([-1, n_frames])
        for pf in gt_picked_fid: 
            ax2.plot((pf, pf), (0, max([x[1] for x in scatter_data])), 'k-' )
        ax2.set_ylim([0, max([x for x in frame_diffs])])
     
        ax3.scatter([x for x in range(1, n_frames)], [x for x in surf_scores], marker = 'x')
        ax3.set_xlabel('Frame ID')
        ax3.set_ylabel('SURF score')
        ax3.set_xlim([-1, n_frames])
        ax3.set_ylim([0, max([x for x in surf_scores]) ])
        for pf in gt_picked_fid: 
            ax3.plot((pf, pf), (0, max([x[1] for x in scatter_data])), 'k-' )

        ax4.scatter([x for x in range(1, n_frames)], [x for x in phashs], marker = 'x')
        ax4.set_xlabel('Frame ID')
        ax4.set_ylabel('PHASH')
        ax4.set_xlim([-1, n_frames])
        ax4.set_ylim([0, max([x for x in phashs])])
        for pf in gt_picked_fid: 
            ax4.plot((pf, pf), (0, max([x[1] for x in scatter_data])), 'k-' )
        
        ax5.scatter([x for x in range(1, n_frames)], [x for x in sobels], marker = 'x')
        ax5.set_xlabel('Frame ID')
        ax5.set_ylabel('Sobel')
        ax5.set_xlim([-1, n_frames])
        ax5.set_ylim([0, max([x for x in sobels])])
        for pf in gt_picked_fid: 
            ax5.plot((pf, pf), (0, max([x[1] for x in scatter_data])), 'k-' )
        
        ax6.scatter([x for x in range(1, n_frames)], [x for x in hist_scores], marker = 'x')
        ax6.set_xlabel('Frame ID')
        ax6.set_ylabel('Color histogram')
        ax6.set_xlim([-1, n_frames])
        ax6.set_ylim([0, max([x for x in hist_scores])])
        for pf in gt_picked_fid: 
            ax6.plot((pf, pf), (0, max([x[1] for x in scatter_data])), 'k-' )

        ax7.scatter([x for x in range(1, n_frames)], [x for x in enc_sizes], marker = 'x')
        ax7.set_xlabel('Frame ID')
        ax7.set_ylabel('Enc sizes')
        ax7.set_xlim([-1, n_frames])
        ax7.set_ylim([0, max([x for x in enc_sizes])])
        for pf in gt_picked_fid: 
            ax7.plot((pf, pf), (0, max([x[1] for x in scatter_data])), 'k-' )
        
        ax8.scatter([x for x in range(1, n_frames)], [x for x in mv_sizes], marker = 'x')
        ax8.set_xlabel('Frame ID')
        ax8.set_ylabel('Motion vector sizes')
        ax8.set_xlim([-1, n_frames])
        ax8.set_ylim([0, max([x for x in mv_sizes])])
        for pf in gt_picked_fid: 
            ax8.plot((pf, pf), (0, max([x[1] for x in scatter_data])), 'k-' )
        
        ax9.scatter([x for x in range(1, n_frames)], [x for x in ave_mv_sizes], marker = 'x')
        ax9.set_xlabel('Frame ID')
        ax9.set_ylabel('Ave Motion vector sizes')
        ax9.set_xlim([-1, n_frames])
        ax9.set_ylim([0, max([x for x in ave_mv_sizes])])
        for pf in gt_picked_fid: 
            ax9.plot((pf, pf), (0, max([x[1] for x in scatter_data])), 'k-' )
        
        ax10.scatter([x for x in range(1, n_frames)], [x for x in illus], marker = 'x')
        ax10.set_xlabel('Frame ID')
        ax10.set_ylabel('Illumination')
        ax10.set_xlim([-1, n_frames])
        ax10.set_ylim([0, max([x for x in illus])])
        for pf in gt_picked_fid: 
            ax10.plot((pf, pf), (0, max([x[1] for x in scatter_data])), 'k-' )
        
        ax11.scatter([x for x in range(1, n_frames)], [x for x in max_mv_sizes], marker = 'x')
        ax11.set_xlabel('Frame ID')
        ax11.set_ylabel('Max Motion vector sizes')
        ax11.set_xlim([-1, n_frames])
        ax11.set_ylim([0, max([x for x in max_mv_sizes])])
        for pf in gt_picked_fid: 
            ax11.plot((pf, pf), (0, max([x[1] for x in scatter_data])), 'k-' )
        
        ax12.scatter([x for x in range(1, n_frames)], [x for x in illu_changes], marker = 'x')
        ax12.set_xlabel('Frame ID')
        ax12.set_ylabel('Changes in illumination')
        ax12.set_xlim([-1, n_frames])
        ax12.set_ylim([0, max([x for x in illu_changes])])
        for pf in gt_picked_fid: 
            ax12.plot((pf, pf), (0, max([x[1] for x in scatter_data])), 'k-' )
        #plt.show()
        plt.savefig('./feature-figs/' + video_name + '.pdf')
        
