from utils import load_all_modules_dict_local
from nlp import *
from wordnet import *
import inflection
import random
import math
import sys

video_names = open('/home/t-yuche/panorama/video_list.txt').read().split()
wptospd = word_pref_to_stopword_pref_dict()

keywords = ['dog', 'cat', 'horse', 'basketball', 'person', 'street', 'guitar', 'snow', 'beach', 'mountain', 'car', 'ball', 'bicycle']

if __name__ == "__main__":

    seed = 0  
    if len(sys.argv) == 2:
        seed = len(sys.argv[1])

    random.seed(seed)

    keyword_index = {}
    for keyword in keywords:
        print keyword        
        video_relevance = {}
        for vid, video_name in enumerate(video_names): 
            print vid, video_name
            if vid == 500:
                break

            rcnn_dict, vgg_dict, dummy, msr_cap_dict, dummy = load_all_modules_dict_local(video_name)     
            video_len =  len(rcnn_dict.keys())
            start_time = random.randint(0, video_len - 1)       
            relevance = []
            total_len = min(start_time + 30 * 30, video_len) - start_time

            for fid in xrange(start_time, min(start_time + 30 * 30, video_len)):
                vgg_merged_tfs = {}
                msr_merged_tfs = {}

                frame_path = '/mnt/frames/' + video_name + '/' + str(fid) + '.jpg'
                stop_words = get_stopwords(0) 

                vgg_data = vgg_dict[frame_path]
                msr_data = msr_cap_dict[frame_path]
                rcnn_data = rcnn_dict[frame_path]
          
 
                ws = {}
                for rcnn_idx, word in enumerate(rcnn_data['pred']['text']):
                    ## the confidence is higher than 10^(-3) and is not background
                    if rcnn_data['pred']['conf'][rcnn_idx] > 0.1 and word.find('__background__') < 0:
                        ws[word] = rcnn_data['pred']['conf'][rcnn_idx]

                for wid, w in enumerate(vgg_data['pred']['text']):
                    w = wptospd[w]
                    prob = (-1)*vgg_data['pred']['conf'][wid]
                    if w in stop_words:
                        continue
                    if w not in vgg_merged_tfs:
                        vgg_merged_tfs[w] = prob
                    else:
                        vgg_merged_tfs[w] += prob


                MSRTOPK = 10
                deno = sum([math.exp(msr_data['words']['prob'][wid]) for wid, w in enumerate(msr_data['words']['text'][:MSRTOPK])])
                for wid, w in enumerate(msr_data['words']['text'][:MSRTOPK]):
                    w = inflection.singularize(w)
                    if w not in stop_words:
                        prob = msr_data['words']['prob'][wid]
                        exp_prob = math.exp(msr_data['words']['prob'][wid])/deno

                        if w not in msr_merged_tfs:
                            msr_merged_tfs[w] = exp_prob
                        else:
                            msr_merged_tfs[w] += exp_prob
                msr_merged_tfs = [x for x in msr_merged_tfs if msr_merged_tfs[x] > 0.3]
                vgg_merged_tfs = [x for x in vgg_merged_tfs if vgg_merged_tfs[x] > 0.3]

                if keyword in msr_merged_tfs or keyword in vgg_merged_tfs or keyword in ws:
                    relevance += [fid]             
                     
            #len(relevance)/total_len
            if len(relevance) > 0:
                
                video_relevance[video_name] = {'score': len(relevance)/(total_len * 1.0), 'obj_pos':relevance, 'video_start': min(relevance), 'video_end': max(relevance)}
            else:
                video_relevance[video_name] = {'score': len(relevance)/(total_len * 1.0), 'obj_pos':relevance, 'video_start': start_time, 'video_end': start_time + total_len}


        keyword_index[keyword] = video_relevance
    import pickle
    with open('index.pickle', 'w') as fh:
        pickle.dump(keyword_index, fh)
