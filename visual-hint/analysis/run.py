import os
import pickle
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

def plot_fig(sel_pairs, x_label, V = 1,NUMBER_BINS =5):
    
    xx = [x[0] for x in sel_pairs]
    xx_hist, xx_edges = np.histogram(xx, bins=NUMBER_BINS) 
    xx_stem_x = stem_x(xx_edges)

    xx_count = [0] * NUMBER_BINS
    for p in sel_pairs:
        if p[1] == V:
            idx = get_bin_idx(xx_edges, p[0])
            xx_count[idx] += 1
    
    print xx_count
    print xx_hist
    print xx_edges, xx_stem_x
    for idx, c in enumerate(xx_count):
        xx_count[idx] /= (xx_hist[idx]  * 1.0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.stem(xx_stem_x, [100 * x for x in xx_count], '-.')
    ax.bar(xx_stem_x, [100 * x for x in xx_count], align = 'center', width = xx_stem_x[1] - xx_stem_x[0])
    ax.axhline(y=50,xmin=0,xmax=100,c="red",linewidth=2,zorder=1, ls='dotted')
    ax.set_ylabel('Percentage of being selected as "Relevant" (%)')
    ax.set_xlabel(x_label)
    ax.set_ylim([0, 100])
    plt.show()

def is_bad_video(bad_videos, v):

    for bv in bad_videos:
        if v.find(bv) >= 0:
            return True

    return False

def show_obj_size(video_info, voted_info):

    bad_videos = open('os_badv').read().split()

    os_sel_pairs = []
    for pair_comp in voted_info:

        va = pair_comp['va']
        vb = pair_comp['vb']
        sel = pair_comp['selection']  

        if va not in video_info.keys() or vb not in video_info.keys() or is_bad_video(bad_videos, va) or is_bad_video(bad_videos, vb):
            continue

        va_os = video_info[va]['obj_region'] 
        vb_os = video_info[vb]['obj_region']
    
        va_dt = video_info[va]['dwell_time'] 
        vb_dt = video_info[vb]['dwell_time']
        
        va_md = video_info[va]['moving_dist'] 
        vb_md = video_info[vb]['moving_dist']
     
    
        value = abs(va_os - vb_os)     
        if abs(va_dt - vb_dt) < 150 and abs(va_md - vb_md) < 1:

            if va_os > vb_os:
                if sel == 'A':
                    os_sel_pairs += [(va_os - vb_os, 1)] 
                else:
                    if value > 0.3 and value < 0.4:
                        print va, vb, value
                    os_sel_pairs += [(va_os - vb_os, 0)] 
            else:
                if sel == 'B':
                    os_sel_pairs += [(vb_os - va_os, 1)] 
                else: 
                    if value > 0.3 and value < 0.4:
                        print vb, va, value
                    os_sel_pairs += [(vb_os - va_os, 0)] 

    #print os_sel_pairs
    with open('os.pickle','wb') as fh:
        pickle.dump(os_sel_pairs, fh)
    #print '-------', os_sel_pairs
    plot_fig(os_sel_pairs, 'Normalized Object Size Difference')

def show_dwell_time(video_info, voted_info):

    bad_videos = open('dt_badv').read().split()

    dt_sel_pairs = []
    for pair_comp in voted_info:

        va = pair_comp['va']
        vb = pair_comp['vb']
        sel = pair_comp['selection']  

        if va not in video_info.keys() or vb not in video_info.keys() or is_bad_video(bad_videos, va) or is_bad_video(bad_videos, vb):
            continue

        va_os = video_info[va]['obj_region'] 
        vb_os = video_info[vb]['obj_region']
    
        va_dt = video_info[va]['dwell_time'] 
        vb_dt = video_info[vb]['dwell_time']
        
        va_md = video_info[va]['moving_dist'] 
        vb_md = video_info[vb]['moving_dist']
     
    
        value = abs(va_dt - vb_dt)     
        if abs(va_os - vb_os) < 0.1 and abs(va_md - vb_md) < 0.1:

            if va_dt > vb_dt:
                if sel == 'A':
                    dt_sel_pairs += [((va_dt - vb_dt)/150., 1)] 
                else:
                    if value > 110 and value < 151:
                        print va, vb, value
                    dt_sel_pairs += [((va_dt - vb_dt)/150., 0)] 
            else:
                if sel == 'B':
                    dt_sel_pairs += [((vb_dt - va_dt)/150., 1)] 
                else: 
                    if value > 110 and value < 151:
                        print vb, va, value
                    dt_sel_pairs += [((vb_dt - va_dt)/150., 0)] 

    with open('dt.pickle', 'wb') as fh:
        pickle.dump(dt_sel_pairs, fh)
    #print os_sel_pairs
    plot_fig(dt_sel_pairs, 'Normalized Dwell Time Difference')

def show_moving_dist(video_info, voted_info):

    bad_videos = []
    bad_videos = open('md_badv').read().split()

    md_sel_pairs = []
    for pair_comp in voted_info:

        va = pair_comp['va']
        vb = pair_comp['vb']
        sel = pair_comp['selection']  

        if va not in video_info.keys() or vb not in video_info.keys() or is_bad_video(bad_videos, va) or is_bad_video(bad_videos, vb):
            continue

        va_os = video_info[va]['obj_region'] 
        vb_os = video_info[vb]['obj_region']
    
        va_dt = video_info[va]['dwell_time'] 
        vb_dt = video_info[vb]['dwell_time']
        
        va_md = video_info[va]['moving_dist'] 
        vb_md = video_info[vb]['moving_dist']
     
    
        value = abs(va_md - vb_md)     
        if abs(va_os - vb_os) < 0.2 and abs(va_dt - vb_dt) < 150:

            if va_md < vb_md:
                if sel == 'A':
                    md_sel_pairs += [(va_md - vb_md, 1)] 
                else:
                    if value > 0.45  and value < 1:
                        print va, vb, value
                    md_sel_pairs += [(va_md - vb_md, 0)] 
            else:
                if sel == 'B':
                    md_sel_pairs += [(vb_md - va_md, 1)] 
                else: 
                    if value > 0.45 and value < 1:
                        print vb, va, value
                    md_sel_pairs += [(vb_md - va_md, 0)] 

    with open('md.pickle', 'wb') as fh:
        pickle.dump(md_sel_pairs, fh)
    #print os_sel_pairs
    plot_fig(md_sel_pairs, 'Normalized Moving Distance Difference', 1)

if __name__ == "__main__":


    LOG_PATH = '/home/t-yuche/visual-hint/analysis/log/'
    video_info = load_video_obj_info('/home/t-yuche/visual-hint/video-seg-info/single_obj.txt')
    
    selection_infos = []
    
    for r in os.listdir(LOG_PATH):
        r_path = os.path.join(LOG_PATH, r)
        for f in os.listdir(r_path):
            f_path = os.path.join(r_path, f)
            selection_infos += [load_turker_video_labels(f_path)]

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


    show_obj_size(video_info, voted_info)
    show_dwell_time(video_info, voted_info)
    show_moving_dist(video_info, voted_info)

    exit()
    dwell_time_list = []
    obj_size_list = []
    moving_dist_list = []
    dwell_time_s_list = []
    obj_size_s_list = []
    moving_dist_s_list = []
    for s in voted_info:
        va = s['va']
        vb = s['vb']

        #if va not in video_info.keys() or vb not in video_info.keys() or vb.find('dog_rmi_gaillard_yfUflij74P4') >= 0 or va.find('dog_rmi_gaillard_yfUflij74P4') >= 0 or vb.find('purdy_the_rabbit_playing_football_PqiTbB-BLOU') >= 0 or va.find('purdy_the_rabbit_playing_football_PqiTbB-BLOU') >= 0 or va.find('puriums_cracked_cell_chlorella__event_our_cat_loves_it_lxjF-iJboLc') >= 0 or vb.find('puriums_cracked_cell_chlorella__event_our_cat_loves_it_lxjF-iJboLc') >= 0:
        if va not in video_info.keys() or vb not in video_info.keys():
            continue
        
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

    print '-----', dwell_time_list    
     
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
        if t[1] == 1:
            idx = get_bin_idx(dt_edges, t[0])
            dt_count[idx] += 1

    for t in obj_size_s_list:
        if t[1] == 1:
            idx = get_bin_idx(os_edges, t[0]) 
            os_count[idx] += 1

    for t in moving_dist_s_list:
        if t[1] == 1:
            idx = get_bin_idx(md_edges, t[0]) 
            md_count[idx] += 1

    print dt_count, dt_hist
    
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
    plt.show()    

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
