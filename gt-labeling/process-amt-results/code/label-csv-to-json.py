from tools.utils import *
import csv
import json
import sys
import os


def createVideoLabelDatabase(video_name, turker_results, suggested_labels, turker_label_folder = "/home/t-yuche/gt-labeling/turker-labels"):

    video_label_folder = os.path.join(turker_label_folder, video_name)
    if not os.path.exists(video_label_folder):
        os.mkdir(video_label_folder)
        
    turker_labels = sorted(turker_results[video_name], key=lambda k: int(k['frame_name'].split('.')[0]))
        
    for turker_label in turker_labels:
        frame_name = turker_label['frame_name']
        selections = turker_label['selections'] # a list
        
        frame_blob = {}
        frame_blob['frame_name'] = frame_name
        frame_blob['gt_labels'] = [] 

        for select in selections:
            
            if select == "_none":
                frame_blob['gt_labels'] += ['_none']
                continue
        
            label_idx = select.split('-')[0]
            starting_label = select.split('-')[-1]

            # extend labeled selection (also consider hierarchy)
            match = False
            matched_labels = []            
            for l in suggested_labels[frame_name][label_idx].split('->'):
                if l == starting_label:
                    match = True
                if match:
                    matched_labels += [l]                 

            frame_blob['gt_labels']  += [matched_labels]

        frame_label_path = os.path.join(video_label_folder, frame_name.split('.')[0] + '.json')
                        
        with open(frame_label_path, 'w') as fh:
            json.dump(frame_blob, fh)
             


if __name__ == "__main__":

     
    amtresults_folder = sys.argv[1]
    result_dict = load_turker_labels(amtresults_folder)
    for video_name in result_dict:
        suggested_labels = load_suggested_labels(video_name)
        createVideoLabelDatabase(video_name, result_dict, suggested_labels)
