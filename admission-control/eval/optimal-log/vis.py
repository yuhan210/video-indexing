import cv2
from utils import *
import math
import sys
import operator
FRAME_FOLDER = '/home/t-yuche/frames'
OPTIMAL_INPUT_FOLDER = '/home/t-yuche/admission-control/eval/optimal-log/optimal-tf'
SERVER_STORAGE_FRAMES = 5 * 30

def getString(tf):
    ss = ''
    for s in tf:
        ss += s + ' ' + str(tf[s]) + '\n'    
    return ss

if __name__ == "__main__":

    wptospd = word_pref_to_stopword_pref_dict()
    stop_words = get_stopwords(1)
    videos = open('/home/t-yuche/video_list.txt').read().split()
    for video_name in videos:
        #if video_name.find('a_bike_ride_as_recorded_by_google_glass_throughglass_vt9i-i545lM') < 0:
        if video_name.find(sys.argv[1]) < 0:
            continue 
        n_frames = get_video_frame_num(video_name) 

        rcnn_dict, vgg_dict, fei_caption_dict, msr_cap_dict, dummy = load_all_modules_dict_local(video_name)

        with open(os.path.join(OPTIMAL_INPUT_FOLDER, video_name + '_' + str(0.5) + '.pickle')) as fh:
            optimal_data = pickle.load(fh)
 

        start_fid = 0
        while True:

            end_fid = min(start_fid + SERVER_STORAGE_FRAMES, n_frames)
            #print 'start:', start_fid, ' end:', end_fid
            if start_fid > n_frames - 1:
                break

            key = str(start_fid) + '-' + str(end_fid)
            optimal_tf = optimal_data[key]
            optimal_tf_s = sorted(optimal_tf.items(), key = operator.itemgetter(1), reverse = True)
            for i in xrange(start_fid, min(start_fid + 30, n_frames)):
                frame = cv2.imread(os.path.join(FRAME_FOLDER, video_name, str(i) + '.jpg'))     
                h,w = frame.shape[:2]
                frame = cv2.resize(frame, (1 * w, 1 * h)) 
                y0, dy = 25, 25
                for sid, s in enumerate(optimal_tf_s): 
                    y = y0 + sid*dy
                    cv2.putText(frame, s[0] + ' ' + str(float("{0:.2f}".format(s[1]))) , (20,y), 2, 1, 255)
                
                frame_path = '/mnt/frames/' + video_name + '/' + str(i) + '.jpg'
                vgg_data = vgg_dict[frame_path]
                y0, dy = 25, 25
                for wid, w in enumerate(vgg_data['pred']['text']):
                    w = wptospd[w]
                    y = y0 + wid*dy
                    prob = (-1)*vgg_data['pred']['conf'][wid]
                    cv2.putText(frame, w + ' ' + str(float("{0:.2f}".format(prob))) , (370,y), 2, 1, 255)
           
                MSRTOPK = 20
                msr_data = msr_cap_dict[frame_path]
                y0, dy = 250, 25
                counter = 0
                # compute prob
                msr_word = {}
                deno = sum([math.exp(msr_data['words']['prob'][wid]) for wid, w in enumerate(msr_data['words']['text'][:MSRTOPK])])
                for wid, w in enumerate(msr_data['words']['text'][:MSRTOPK]):
                    w = inflection.singularize(w)
                    if w not in stop_words:
                        prob = msr_data['words']['prob'][wid]
                        exp_prob = math.exp(msr_data['words']['prob'][wid])/deno
                        y = y0 + counter*dy
                        counter += 1
                        cv2.putText(frame, w + ' ' + str(float("{0:.2f}".format(prob))) + ' ' + str(float("{0:.2f}".format(exp_prob))) , (20,y), 2, 1, 255)
                    else: 
                        prob = msr_data['words']['prob'][wid]
                        exp_prob = math.exp(msr_data['words']['prob'][wid])/deno
                        y = y0 + counter*dy
                        counter += 1
                        cv2.putText(frame, w + ' ' + str(float("{0:.2f}".format(prob))) + ' ' + str(float("{0:.2f}".format(exp_prob))) , (20,y), 2, 1, 0)
                            
                            
                img_h,img_w = frame.shape[:2]
                cv2.putText(frame, str(i), (img_w - 30,img_h - 30), 2, 1, 0)
                #cv2.imwrite('./tmp/' +  str(i) + '.jpg', frame)                 
                cv2.imshow(video_name , frame)
                cv2.waitKey(-1)
            start_fid += 1 * 30
