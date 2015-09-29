from utils import *
from wordnet import *
import inflection


def check_trace_match(trace, gt_words):

if __name__ == "__main__":

    with open('/mnt/video_list.txt') as fh:
        videos = fh.read().split()
 
    wptownid = word_prefix_to_wnid_dict()
    wnidtotrace = wnid_traces_dict()
    rcnn_sel = []
    rcnn_unsel = []
    vgg_sel = []
    vgg_unsel = []
    fei_sel = []
    fei_unsel = []
    msr_sel = []
    msr_unsel = []
    for video in videos:
        #load gt 
        turker_labels = load_video_processed_turker('/mnt/tags/turker-all', video) 
    
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

                vgg_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == frame_id, _vgg_data) 
                fei_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == frame_id, _fei_caption_data) 
                msr_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) == frame_id, _msr_cap_data) 
                rcnn_data = filter(lambda x: int(x['image_path'].split('/')[-1].split('.')[0]) == frame_id, _rcnn_data) 
                
               
                for rcnn_idx, word in enumerate(rcnn_data['pred']['text']): 
                    prob = rcnn_data['pref']['conf'][rcnn_idx] 
                    word = inflection.singularize(word)
                
                    if word in gt_words: 
                        rcnn_sel += [prob]    
                    else:
                        rcnn_unsel += [prob]

                for vgg_idx, word in enumerate(vgg_data['pred']['text']):
                    # get ancestors
                    wnid = wptownid[word] 
                    trace = wnidtotrace[wnid]
                     

                for msr_data['words']['text'] 
                                   
