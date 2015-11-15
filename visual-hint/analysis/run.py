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
            continue
            #print row

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

def stem_x(edges):
    bin_num = len(edges) -1
    x = []
    for i in xrange(len(edges)):
        if i == len(edges) - 1:
            break
    
        x+= [(edges[i]  + edges[i+1])/(2 * 1.0)]
        

    return x

def get_bin_idx(edges, value):
    for i in xrange(len(edges) -1):
        if value >= edges[i] and value < edges[i+1]:
            return i

    if value == edges[-1]:
        return len(edges) -2
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
    dwell_time_s_list = []
    obj_size_s_list = []
    moving_dist_s_list = []
    for s in voted_info:
        va = s['va']
        vb = s['vb']

        
        t1 = video_info[va]['dwell_time'] - video_info[vb]['dwell_time']
        r1 = video_info[va]['obj_region'] - video_info[vb]['obj_region']
        m1 = video_info[va]['moving_dist'] - video_info[vb]['moving_dist']
        if s['selection']  == 'A':
            dwell_time_s_list += [(t1, 1)]
            obj_size_s_list += [(r1, 1)]
            moving_dist_s_list += [(m1, 1)]
        else:
            dwell_time_s_list += [(t1, 0)]
            obj_size_s_list += [(r1, 0)]
            moving_dist_s_list += [(m1, 0)]

        t2 = video_info[vb]['dwell_time'] - video_info[va]['dwell_time']
        r2 = video_info[vb]['obj_region'] - video_info[va]['obj_region']
        m2 = video_info[vb]['moving_dist'] - video_info[va]['moving_dist']
        if s['selection']  == 'B':
            dwell_time_s_list += [(t2, 1)]
            obj_size_s_list += [(r2, 1)]
            moving_dist_s_list += [(m2, 1)]
        else:
            dwell_time_s_list += [(t2, 0)]
            obj_size_s_list += [(r2, 0)]
            moving_dist_s_list += [(m2, 0)]
    


        if s['selection'] == 'A':
            vr = s['va'] 
            vlr = s['vb']

        elif s['selection'] == 'B':
            vr = s['vb'] 
            vlr = s['va']


                
        dwell_time_list += [video_info[vr]['dwell_time'] - video_info[vlr]['dwell_time']]
        obj_size_list += [video_info[vr]['obj_region'] - video_info[vlr]['obj_region']]
        moving_dist_list += [video_info[vr]['moving_dist'] - video_info[vlr]['moving_dist']]

         
    dts = [x[0] for x in dwell_time_s_list]
    oss = [x[0] for x in obj_size_s_list] 
    mds = [x[0] for x in moving_dist_s_list]    

    NUMBER_BINS = 15
    dt_hist, dt_edges = np.histogram(dts, bins=NUMBER_BINS) 
    os_hist, os_edges = np.histogram(oss, bins=NUMBER_BINS) 
    md_hist, md_edges = np.histogram(mds, bins=NUMBER_BINS) 

    
    dt_stem_x = stem_x(dt_edges)
    os_stem_x = stem_x(os_edges)
    md_stem_x = stem_x(md_edges)
    dt_count = [0] * NUMBER_BINS
    os_count = [0] * NUMBER_BINS
    md_count = [0] * NUMBER_BINS
    for t in dwell_time_s_list:
        if t[1] > 0:
            idx = get_bin_idx(dt_edges, t[0])
            dt_count[idx] += 1

    for t in obj_size_s_list:
        if t[1] > 0:
            idx = get_bin_idx(os_edges, t[0]) 
            os_count[idx] += 1

    for t in moving_dist_s_list:
        if t[1] > 0:
            idx = get_bin_idx(md_edges, t[0]) 
            md_count[idx] += 1

    
    # normalize
    for idx, c in enumerate(dt_count):
        dt_count[idx] /= (dt_hist[idx] * 1.0) 

    for idx, c in enumerate(os_count):
        os_count[idx] /= (os_hist[idx] * 1.0)

    for idx, c in enumerate(md_count):
        md_count[idx] /= (md_hist[idx]  * 1.0)

    fig = plt.figure(1)
    ax = fig.add_subplot(131)
    ax.stem(dt_stem_x, [100 * x for x in dt_count], '-.')
    ax.set_ylim([0, 100])
    ax.set_xlabel('Normalized Dwell \nTime Difference')
    ax.set_ylabel('Porbability of Selected as "Relevant" (%)')
    ax.set_title('Dwell time') 

    ax = fig.add_subplot(132)
    ax.stem(os_stem_x, [100 * x for x in os_count] , '-.')
    ax.set_ylim([0, 100])
    ax.set_xlabel('Normalized Object \nSize Difference')
    ax.set_title('Object size') 

    ax = fig.add_subplot(133)
    ax.stem(md_stem_x, [100 * x for x in md_count], '-.')
    ax.set_ylim([0, 100])
    ax.set_xlabel('Normalized Object \nMoving Distance Difference')
    ax.set_title('Moving distance') 
    fig.savefig('vis-hint-obser.pdf', bbox_inches = 'tight')
    
    fig = plt.figure(2)
    ax = fig.add_subplot(311)
    ax.plot(sorted(dwell_time_list), np.arange(len(dwell_time_list))/float(len(dwell_time_list)))
    ax.set_title('Dwell time') 
    ax = fig.add_subplot(312)
    ax.plot(sorted(obj_size_list), np.arange(len(obj_size_list))/float(len(obj_size_list)))
    ax.set_title('Object size')
    ax = fig.add_subplot(313)
    ax.plot(sorted(moving_dist_list), np.arange(len(moving_dist_list))/float(len(moving_dist_list)))
    ax.set_title('Moving distance') 
    #plt.show() 
