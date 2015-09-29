from utils import *
from wordnet import *
from nlp import *
import numpy as np
import inflection
import matplotlib
import matplotlib.pyplot as plt
try:
    plt.style.use('ggplot')
except:
    pass



def _plot(msr_sel, msr_unsel, vgg_sel, vgg_unsel, fei_sel, fei_unsel, rcnn_sel, rcnn_unsel):

    msr_sel_arr = np.asarray(msr_sel)
    msr_unsel_arr = np.asarray(msr_unsel)
    vgg_sel_arr = np.asarray(vgg_sel)
    vgg_unsel_arr = np.asarray(vgg_unsel)
    fei_sel_arr = np.asarray(fei_sel)
    fei_unsel_arr = np.asarray(fei_unsel)
    rcnn_sel_arr = np.asarray(rcnn_sel)
    rcnn_unsel_arr = np.asarray(rcnn_unsel)
    # plot histogram
    bin_num = 500
    
    plt.figure(1)
    plt.title('MSR captioning')
    shist, sbins = np.histogram(msr_sel_arr, bins = bin_num, density = True)
    swidth = sbins[1] - sbins[0]
    plt.bar(sbins[:-1], shist * swidth, swidth, facecolor = 'red', alpha = 0.7, label = 'MSR selected')
    
    uhist, ubins = np.histogram(msr_unsel_arr, bins = bin_num, density = True)
    uwidth = ubins[1] - ubins[0]
    plt.bar(ubins[:-1], uhist * uwidth, uwidth, facecolor = 'blue', alpha = 0.7, label = 'MSR unselected')

    plt.legend(loc='upper right')
 
    plt.figure(2)
    plt.title('VGG recognition')
    shist, sbins = np.histogram(vgg_sel_arr, bins = bin_num, density = True)
    swidth = sbins[1] - sbins[0]
    plt.bar(sbins[:-1], shist * swidth, swidth, facecolor = 'red', alpha = 0.7, label = 'VGG selected')
    
    uhist, ubins = np.histogram(vgg_unsel_arr, bins = bin_num, density = True)
    uwidth = ubins[1] - ubins[0]
    plt.bar(ubins[:-1], uhist * uwidth, uwidth, facecolor = 'blue', alpha = 0.7, label = 'VGG unselected')
    #sw = np.ones_like(vgg_sel_arr)/float(len(vgg_sel_arr))
    #uw = np.ones_like(vgg_unsel_arr)/float(len(vgg_unsel_arr))
    #plt.hist(vgg_sel_arr,  weights = sw,  alpha = 0.7, label = 'VGG selected')
    #plt.hist(vgg_unsel_arr, weights = uw , alpha = 0.7, label = 'VGG unselected')
    plt.legend(loc='upper right')
 
    plt.figure(3)
    plt.title('Fei-Fei captioning')
    shist, sbins = np.histogram(fei_sel_arr, bins = bin_num, density = True)
    swidth = sbins[1] - sbins[0]
    plt.bar(sbins[:-1], shist * swidth, swidth, facecolor = 'red', alpha = 0.7, label = 'Fei selected')
    
    uhist, ubins = np.histogram(fei_unsel_arr, bins = bin_num, density = True)
    uwidth = ubins[1] - ubins[0]
    plt.bar(ubins[:-1], uhist * uwidth, uwidth, facecolor = 'blue', alpha = 0.7, label = 'Fei unselected')
    
    #sw = np.ones_like(fei_sel_arr)/float(len(fei_sel_arr))
    #uw = np.ones_like(fei_unsel_arr)/float(len(fei_unsel_arr))
    #plt.hist(fei_sel_arr, weights = sw,  alpha = 0.7, label = 'Fei selected')
    #plt.hist(fei_unsel_arr, weights = uw , alpha = 0.7, label = 'Fei unselected')
    plt.legend(loc='upper right')

    plt.figure(4)
    plt.title('RCNN')
    shist, sbins = np.histogram(rcnn_sel_arr, bins = bin_num, density = True)
    swidth = sbins[1] - sbins[0]
    plt.bar(sbins[:-1], shist * swidth, swidth, facecolor = 'red', alpha = 0.7, label = 'RCNN selected')
    
    uhist, ubins = np.histogram(rcnn_unsel_arr, bins = bin_num, density = True)
    uwidth = ubins[1] - ubins[0]
    plt.bar(ubins[:-1], uhist * uwidth, uwidth, facecolor = 'blue', alpha = 0.7, label = 'RCNN unselected')
    
    #sw = np.ones_like(rcnn_sel_arr)/float(len(rcnn_sel_arr))
    #uw = np.ones_like(rcnn_unsel_arr)/float(len(rcnn_unsel_arr))
    #plt.hist(rcnn_sel_arr, weights = sw,  alpha = 0.7, label = 'RCNN selected')
    #plt.hist(rcnn_unsel_arr, weights = uw , alpha = 0.7, label = 'RCNN unselected')
    plt.legend(loc='upper right')
    plt.show()

def check_trace_match(trace, gt_words):

    for pl in trace:
        for p in pl:
            for gt_word in gt_words:
                if p.find(gt_word) == 0:
                    return True

    return False

if __name__ == "__main__":

    with open('/mnt/video_list.txt') as fh:
        videos = fh.read().split()
 
    wptownid = word_prefix_to_wnid_dict()
    wnidtotrace = wnid_traces_dict()
    stop_words = get_stopwords()
    rcnn_sel = []
    rcnn_unsel = []
    vgg_sel = []
    vgg_unsel = []
    fei_sel = []
    fei_unsel = []
    msr_sel = []
    msr_unsel = []
    for vid, video_name in enumerate(videos):
        print vid, video_name
        #load gt 
        turker_labels = load_video_processed_turker('/mnt/tags/turker-all', video_name) 
    
        if turker_labels != None: 
         
            # load tags
            _vgg_data = load_video_recog('/mnt/tags/vgg-classify-all', video_name)
            _fei_caption_data = load_video_caption('/mnt/tags/fei-caption-all', video_name)
            _msr_cap_data = load_video_msr_caption('/mnt/tags/msr-caption-all', video_name)
            _rcnn_data = load_video_rcnn('/mnt/tags/rcnn-info-all', video_name)

            
            for turker_label in turker_labels:
                frame_name = turker_label['frame_name']
                gt_words = turker_label['gt_labels']
                frame_id = int(frame_name.split('.')[0])

                vgg_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == frame_id, _vgg_data)[0] 
                fei_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == frame_id, _fei_caption_data)[0]
                msr_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == frame_id, _msr_cap_data)[0]
                rcnn_data = filter(lambda x: int(x['image_path'].split('/')[-1].split('.')[0]) == frame_id, _rcnn_data)[0]
                
               
                for rcnn_idx, word in enumerate(rcnn_data['pred']['text']): 
                    prob = rcnn_data['pred']['conf'][rcnn_idx] 
                    word = inflection.singularize(word)
                
                    if word in gt_words: 
                        rcnn_sel += [prob]    
                    else:
                        rcnn_unsel += [prob]

                for vgg_idx, word in enumerate(vgg_data['pred']['text']):
                    # get ancestors
                    prob = (vgg_data['pred']['conf'][vgg_idx]) * (-1)
                    wnid = wptownid[word] 
                    trace = wnidtotrace[wnid]
                    is_match = check_trace_match(trace, gt_words)
                    if is_match:
                        vgg_sel += [prob]
                    else:
                        vgg_unsel += [prob]
           
                for msr_idx, word in enumerate(msr_data['words']['text']): 
                    word = inflection.singularize(word)
                    prob = msr_data['words']['prob'][msr_idx]
                    if word in stop_words:
                        continue 
                    if word in gt_words:
                        msr_sel += [prob]
                    else:
                        msr_unsel += [prob] 

                for fei_idx, s in enumerate(fei_data['candidate']['text']):
                    prob = fei_data['candidate']['logprob'][fei_idx]
                    words = s.split() 
                    for word in words:
                        if word in stop_words:
                            continue
                        if word in gt_words:
                            fei_sel += [prob] 
                        else:
                            fei_unsel += [prob]

        if vid % 100 == 0:
            _plot(msr_sel, msr_unsel, vgg_sel, vgg_unsel, fei_sel, fei_unsel, rcnn_sel, rcnn_unsel)


    _plot(msr_sel, msr_unsel, vgg_sel, vgg_unsel, fei_sel, fei_unsel, rcnn_sel, rcnn_unsel)
