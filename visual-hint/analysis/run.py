import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
try:
    plt.style.use('ggplot')
except:
    pass


def load_turker_video_labels(file_name):
    csv_file = open(file_name)
    csv_reader = csv.DictReader(csv_file, delimiter="\t") 
   
    selection_info = {}
    for row in csv_reader:
        video_name_a = row['Answer.video_name_a']
        video_name_b = row['Answer.video_name_b']
        selection = row['Answer.selected_value'] 
        if len(selection) == 0:
            print row

        key = video_name_a + '_' + video_name_b
        selection_info[key] = {'va': video_name_a, 'vb':video_name_b, 'selection': selection}

    return selection_info

def load_video_obj_info(file_name):

    lines = [x.strip() for x in open(file_name).readlines()]
    video_info = {}

    for line in lines:
        segs = line.split(',')
        video_name = segs[0]
        start_fid = segs[1]
        end_fid = segs[2]

        obj_num = int(segs[3])
        dwell_time = int(segs[4])
        obj_size = (float(segs[5].split('(')[-1]),  float(segs[6].split(')')[0]))
        moving_dist = float(segs[7])
        turker_select_num = int(segs[8])
        turker_shown_num = int(segs[9])

        key = video_name + '_' + start_fid + '_' + end_fid + '.mp4' 

        video_info[key] = {'obj_num': obj_num, 'dwell_time': dwell_time, 'obj_size': obj_size, 'obj_region': obj_size[0] * obj_size[1], 'moving_dist': moving_dist, 'turker_select_num': turker_select_num, 'turker_shown_num': turker_shown_num}

    return video_info

def maj_vote(l):
    d = {'A': 0, 'B': 0}
    for i in l:
        d[i] += 1

    if d['A'] > d['B']:
        return 'A'
    else:
        return 'B'

if __name__ == "__main__":

    video_info = load_video_obj_info('/home/t-yuche/visual-hint/video-seg-info/single_obj.txt')
    
    selection_infos = []
    
    selection_infos += [load_turker_video_labels('/home/t-yuche/visual-hint/analysis/log/video_label.results-1')]
    selection_infos += [load_turker_video_labels('/home/t-yuche/visual-hint/analysis/log/video_label.results-2')]
    selection_infos += [load_turker_video_labels('/home/t-yuche/visual-hint/analysis/log/video_label.results-3')]

    selection_info = {}
    # merge selection_info
    for ss in selection_infos:
        for key in ss:
            if key not in selection_info.keys():
                selection_info[key] = [ss[key]['selection']]
            else:
                selection_info[key] += [ss[key]['selection']]

    voted_info = []
    for key in selection_info:
        if len(key.split('.mp4_')) != 2:
            continue
        video_name_a = key.split('.mp4_')[0] + '.mp4' 
        video_name_b = key.split('.mp4_')[1] 
        sel = maj_vote(selection_info[key])
        voted_info += [{'selection': sel, 'va': video_name_a, 'vb': video_name_b}]
 
    dwell_time_list = []
    obj_size_list = []
    moving_dist_list = []
    for s in voted_info:
        if s['selection'] == 'A':
            vr = s['va'] 
            vlr = s['vb']

        elif s['selection'] == 'B':
            vr = s['vb'] 
            vlr = s['va']

        
        dwell_time_list += [[video_info[vr]['dwell_time'] - video_info[vlr]['dwell_time']]]
        obj_size_list += [video_info[vr]['obj_region'] - video_info[vlr]['obj_region']]
        moving_dist_list += [video_info[vr]['moving_dist'] - video_info[vlr]['moving_dist']]

    plt.figure(1)
    plt.plot(sorted(dwell_time_list), np.arange(len(dwell_time_list))/float(len(dwell_time_list)))
    plt.title('Dwell time') 
    plt.figure(2)
    plt.plot(sorted(obj_size_list), np.arange(len(obj_size_list))/float(len(obj_size_list)))
    plt.title('Object size')
    plt.figure(3) 
    plt.plot(sorted(moving_dist_list), np.arange(len(moving_dist_list))/float(len(moving_dist_list)))
    plt.title('Moving distance') 
    plt.show() 
