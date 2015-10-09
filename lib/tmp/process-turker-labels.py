from utils import *
import inflection
import json

"""
Combine turker labels into one file
"""

if __name__ == "__main__":
    
    with open('/mnt/video_list.txt') as fh:
        videos = fh.read().split()

    for video in videos:
        #print video
        amt_labels = load_video_turker('/mnt/turker-labels', video)     
        
        if len(amt_labels) == 0:
            print 'No turker label for', video
            continue

        video_labels = []
        for key in amt_labels:
            video_label = {}
            all_gt_words = []
            for gl in amt_labels[key]:
                for g in gl:
                    if not g.find('none') >= 0:
                        gt_word = inflection.singularize(g.split('-')[-1])
                        if gt_word not in all_gt_words:
                            all_gt_words += [gt_word]

            video_label['frame_name'] = key
            video_label['gt_labels'] = all_gt_words
            video_labels += [video_label]

        video_labels = sorted(video_labels, key=lambda x: int(x['frame_name'].split('.')[0]))  
        with open(os.path.join('/mnt/tags/turker-all', video + '.json'), 'w') as fh:
            json.dump(video_labels, fh)

